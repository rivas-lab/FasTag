##################################################################################
# Notes: Plotting for summary information about the notes, words, and icd9 codes.
# TODO:
# 1) Need to take into account the new vocabulary.
##################################################################################

library(data.table)
library(dplyr)
library(lubridate)
library(ggplot2)
library(stringr)
icd9NotesDataTable = tbl_df(fread('data/icd9NotesDataTable.csv'))



##################################################################################
# Distribution of icd9 code counts
##################################################################################
allICD9Codes = sapply(icd9NotesDataTable$ICD9_CODE, function(x) strsplit(x, '-'))
allICD9Codes = unlist(allICD9Codes)
# allICD9Codes = tbl_df(data.frame(allICD9Codes, stringsAsFactors = FALSE))
icd9CodeCounts = table(allICD9Codes)
icd9CodeCounts = tbl_df(data.frame(code = names(icd9CodeCounts), count = icd9CodeCounts, stringsAsFactors = FALSE))
icd9CodeCounts = select(icd9CodeCounts, code, count.Freq)
ggplot(icd9CodeCounts, aes(x = count.Freq, group = 1)) + geom_step(aes(y=..y..),color='#78a5a3',stat="ecdf") + 
  # scale_colour_manual(values =  c('#78a5a3')) +
  scale_x_continuous(limits = c(0, 1000), expand = c(0, 0)) +
  scale_y_continuous(expand = c(0, 0)) + 
  labs(x = "Number of Labels", y = 'Empirical Cumulative Distribution', title = 'ICD-9 Label Distribution') + 
  theme_bw() + theme(plot.title = element_text(hjust = 0.5, size = 18), axis.title=element_text(size=18), 
                     legend.text = element_text(size = 15), legend.title = element_text(size = 15))
# x <- data.frame(A=replicate(200,sample(c("a","b","c"),1)),X=rnorm(200))
# ggplot(x,aes(x=X,color=A)) + geom_step(aes(y=..y..),stat="ecdf")
ggsave(filename = 'src/plotting/dataExploration/icd9LabelDist.png')

icd9CodeCounts = arrange(icd9CodeCounts, desc(count.Freq))
icd9CodeCounts = mutate(icd9CodeCounts, cumsum(count.Freq)/sum(count.Freq))


##################################################################################
# Distribution of top level icd9 code counts
##################################################################################
allICD9Codes = sapply(icd9NotesDataTable$V9, function(x) strsplit(x, '-'))
allICD9Codes = unlist(allICD9Codes)
allICD9Codes = gsub(pattern = "cat:", replacement = '', allICD9Codes)
allICD9Codes = as.integer(allICD9Codes)
allICD9Codes = tbl_df(data.frame(code = allICD9Codes))

ggplot(allICD9Codes, aes(code)) + geom_bar(col="#e1b16a",
                                           fill="#ce5a57") + 
  labs(x = "Top Level ICD-9 Codes", y = 'Count', title = 'ICD-9 Label Distribution') + 
  theme_bw() + theme(plot.title = element_text(hjust = 0.5, size = 18), axis.title=element_text(size=18), 
                     legend.text = element_text(size = 15), legend.title = element_text(size = 15)) +
  scale_x_continuous(breaks = seq(0, 19, by = 1), expand = c(.01, .01))
ggsave(filename = 'src/plotting/dataExploration/topLevelIcd9LabelDist.png')

##################################################################################
# Distribution of icd9 codes per admission
##################################################################################
admissionICD9Count = select(icd9NotesDataTable, HADM_ID, ICD9_CODE)

admissionICD9Count = mutate(group_by(admissionICD9Count, HADM_ID), icd9Count = length(strsplit(ICD9_CODE, '-')[[1]])) 
admissionICD9Count
meanLineName = paste('Mean(', toString(round(mean(admissionICD9Count$icd9Count), digits = 2)),')', sep = '')
medLineName = paste('Median(', toString(round(median(admissionICD9Count$icd9Count), digits = 2)),')', sep = '')
ggplot(data=admissionICD9Count, aes(icd9Count)) + geom_histogram(binwidth = 2, col="#e1b16a",
                                                                 fill="#ce5a57",
                                                                 alpha = .8) +
  geom_vline(aes(xintercept = mean(icd9Count), colour="mean", alpha = 'mean', linetype = 'mean'), show.legend = TRUE) +
  geom_vline(aes(xintercept = median(icd9Count), colour="median", alpha = 'median', 'linetype' = 'median'), show.legend = TRUE) +
  scale_x_continuous(expand = c(0, 0)) +
  scale_y_continuous(expand = c(0, 0)) +
  labs(title = 'ICD-9 Counts Per Admission', y = 'Count', x = 'Number of ICD-9 Codes') + 
  scale_colour_manual(name="Legend", values = c("mean" = "#78a5a3", "median" = "#444c5c")) +
  scale_linetype_manual(name="Legend", values = c("mean" = "dashed", "median" = "dashed")) +
  scale_alpha_manual(name="Legend", values = c('mean' = 1, 'median' = 0.6)) +
  theme_bw() + theme(plot.title = element_text(hjust = 0.5, size = 18), axis.title=element_text(size=18), 
                     legend.text = element_text(size = 15), legend.title = element_text(size = 15))
ggsave(filename = 'src/plotting/dataExploration/icd9CountHist.png')
print(meanLineName)
print(medLineName)




##################################################################################
# Distribution of note length
##################################################################################
countWords = function(note, vocab = c()){
  # print('hey')
  # print(note)
  return(sum(str_count(note, vocab)))
}
vocab = c('the', 'but')
admissionWordCount = select(icd9NotesDataTable, HADM_ID, TEXT)
admissionWordCount = mutate(group_by(admissionWordCount, HADM_ID), wordCount = countWords(TEXT, vocab))

ggplot(data=admissionWordCount, aes(wordCount)) + geom_histogram(binwidth = 10, col="#e1b16a",
                                                                 fill="#ce5a57",
                                                                 alpha = .8) +
  geom_vline(aes(xintercept = mean(wordCount), colour="mean", alpha = 'mean', linetype = 'mean'), show.legend = TRUE) +
  geom_vline(aes(xintercept = median(wordCount), colour="median", alpha = 'median', 'linetype' = 'median'), show.legend = TRUE) +
  scale_x_continuous(expand = c(0, 0)) +
  scale_y_continuous(expand = c(0, 0)) +
  labs(title = 'Note Length Per Admission', y = 'Count', x = 'Number of Vocabulary Words') + 
  scale_colour_manual(name="Legend", values = c("mean" = "#78a5a3", "median" = "#444c5c")) +
  scale_linetype_manual(name="Legend", values = c("mean" = "dashed", "median" = "dashed")) +
  scale_alpha_manual(name="Legend", values = c('mean' = 1, 'median' = 0.6)) +
  theme_bw() + theme(plot.title = element_text(hjust = 0.5, size = 18), axis.title=element_text(size=18), 
                     legend.text = element_text(size = 15), legend.title = element_text(size = 15))
ggsave(filename = 'src/plotting/dataExploration/noteLengthByAdmission.png')
print(summary(admissionWordCount$wordCount))







##################################################################################
# Distribution of admission counts
##################################################################################
admissionCounts = select(icd9NotesDataTable, HADM_ID, SUBJECT_ID)
admissionCounts = group_by(admissionCounts, SUBJECT_ID)
admissionCounts = summarise(admissionCounts, admissionCounts = n())


ggplot(data=admissionCounts, aes(admissionCounts)) + geom_histogram(binwidth = 2, col="#e1b16a",
                                                                 fill="#ce5a57",
                                                                 alpha = .8) +
  scale_x_continuous(expand = c(0, 0)) +
  scale_y_continuous(expand = c(0, 0)) +
  labs(title = 'Number of Admissions Per Patient', y = 'Count', x = 'Number of Admissions') + 
  theme_bw() + theme(plot.title = element_text(hjust = 0.5, size = 18), axis.title=element_text(size=18), 
                     legend.text = element_text(size = 15), legend.title = element_text(size = 15))
ggsave(filename = 'src/plotting/dataExploration/numberOfAdmissionsByPatient.png')

summary(admissionCounts$admissionCounts)

##################################################################################
# Summary stats. N patients, N admissions, etc 
##################################################################################
print('Number of admissions')
print(length(icd9NotesDataTable$HADM_ID))
print('****************************')


print('Number of Patients')
print(length(unique(icd9NotesDataTable$SUBJECT_ID)))
print('****************************')

print('ICD-9 Count by Admission Summary')
print(summary(admissionICD9Count$icd9Count))
print('****************************')


print('Note Word Count by Admission Summary')
print(summary(admissionWordCount$wordCount))
print('****************************')


print('Number of Admissions by Patient')
print(summary(admissionCounts$admissionCounts))
print(round(table(admissionCounts$admissionCounts)/sum(admissionCounts$admissionCounts)*100, digits = 4))
print('****************************')

