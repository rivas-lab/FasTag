##################################################################################
# Notes: This is code for reading in the raw MIMIC data and processing it into
# something that can be worked with in the pipeline. 
# TODO:
  # 1) Process text by removing stop words and creating a vocab
##################################################################################





library(data.table)
library(dplyr)
library(lubridate)
Sys.setenv(TZ='UTC')




##################################################################################
# Load in Data
##################################################################################
# notes = tbl_df(fread('data/notesWithoutText.csv'))# For testing because smaller.
notes = tbl_df(fread('data/NOTEEVENTS.csv'))# actual file use when testing is done

# Please check out the below rows in the clinical notes. The addendum one has information
# like DISCHARGE DIAGNOSES which gives away pretty easily if kept in the data.
# notes = filter(notes, ROW_ID %in% c(8772, 55972))
diagnoses = tbl_df(fread('data/DIAGNOSES_ICD.csv'))



##################################################################################
# Collapse all ICD9 codes for each admission. Order might not be preserved
##################################################################################
diagnoses = select(diagnoses, HADM_ID, SUBJECT_ID, ICD9_CODE)
diagnoses = group_by_(diagnoses, .dots=c("HADM_ID","SUBJECT_ID"))
diagnoses = summarise_each(diagnoses, funs(paste(., collapse = "-")))
diagnoses = ungroup(diagnoses)



##################################################################################
# Filter Notes
##################################################################################
notes = filter(notes, CATEGORY == 'Discharge summary')
# there are duplicated HADM_IDs so there are multiple discharge notes
# there must be a smart way to remove the duplicates, but for now
# might just take the one with the data which comes first

notes = select(notes, HADM_ID, CHARTDATE, DESCRIPTION, TEXT)
notes = mutate(notes, CHARTDATE = as.POSIXct(CHARTDATE))
notes = group_by_(notes, .dots=c("HADM_ID"))
notes = filter(notes, CHARTDATE == min(CHARTDATE))# take the first one because idk why. Have to understand why there are more than one discharge summaries
notes = filter(notes, !duplicated(HADM_ID))# Still have duplicats on the same day. For right now removing can do better later.

##################################################################################
# Create note and ICD9 code matrix
##################################################################################
notes = ungroup(notes); diagnoses = ungroup(diagnoses)
icd9NotesDataTable = right_join(x = diagnoses, y = notes, by = 'HADM_ID')


##################################################################################
# Extra filtering
##################################################################################

# filter out any words not needed
# filter out and icd9 codes not needed




##################################################################################
# Split into Training and Validation
# consider smarter splits which consider dates and admission IDs so you don't have 
# any admissions in train and test
##################################################################################
trainingFrac = 0.75
trainingIdxs = sample.int(n = floor(nrow(icd9NotesDataTable)*trainingFrac), replace = FALSE)
icd9NotesDataTable_train = icd9NotesDataTable[trainingIdxs,]
icd9NotesDataTable_valid = icd9NotesDataTable[-trainingIdxs,]

##################################################################################
# Write to file
##################################################################################
write.csv(icd9NotesDataTable, 'data/icd9NotesDataTable.csv')
write.csv(icd9NotesDataTable_train, 'data/icd9NotesDataTable_train.csv')
write.csv(icd9NotesDataTable_valid, 'data/icd9NotesDataTable_valid.csv')




##################################################################################
# Write filesi n a way that is easily read by python
##################################################################################
# open file here
for(lineIdx in 1:nrow(icd9NotesDataTable_train)){
  line = icd9NotesDataTable_train[lineIdx,]
  # for each line 
    # write one word
}




