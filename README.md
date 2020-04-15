# FasTag

## Description
FasTag is a text classification document for the rapid triaging of clinical documents.

## Citation
For citation and more information refer to:

FasTag application:
>[A. Lopez Pineda, O. J. Bear Don't Walk IV, G. R. Venkataraman, A. M. Zehnder et al. (2019) "FasTag: automatic text classification of unstructured medical narratives".](https://www.biorxiv.org/content/10.1101/429720v2) (under review) 

FasTag initial implementation:
>[S. Ayyar, O. J. Bear Don't Walk IV (2017). "Tagging Patient Notes with ICD-9 Codes". Technical Report. Stanford University.](https://github.com/rivas-lab/clinicalNoteTagger/blob/master/obdw4_sa_cd224n.pdf)

## To Train on Default Dataset (MIMIC)

### Setup Server with TensorFlow-GPU

Make sure that you have a GPU with at least 50GB RAM for training this neural network. Otherwise, the computations become intractable.

We have used Azure NV6 and Amazon P2 (with Deep Learning AMI) in the past. The benefit of the Deep Learning AMI is that it comes with a version of TensorFlow-GPU installed with all of the necessary NVIDIA/CUDA GPU driver prerequisites also functioning.

### Run Preprocessing

- Clone the repository to both your local machine and the compute instance. The script `createAdmissionNoteTable.R` from `src/textPreprocessing` is meant to process the MIMIC database and extract only the relevant information (notes and corresponding ICD-9 codes).

- Gain access to the MIMIC-III database, and then [download the files labeled DIAGNOSES_ICD.csv and NOTEEVENTS.csv here](https://physionet.org/works/MIMICIIIClinicalDatabase/files/).

- Replace the absolute paths to these two `.csv`s in lines 25 and 30 in the R script, and run the script. You should end up with three files: `icd9NotesDataTable_train.csv`, `icd9NotesDataTable_valid.csv`, and `icd9NotesDataTable.csv`. These will be your training and validation sets.

- `scp` these three files to the server by running these commands:

`scp path/to/icd9NotesDataTable_train.csv username@domain:~/clinicalNoteTagger/data/`


`scp path/to/icd9NotesDataTable_valid.csv username@domain:~/clinicalNoteTagger/data/`


`scp path/to/icd9NotesDataTable.csv username@domain:~/clinicalNoteTagger/data/`

- In a new terminal window, `ssh username@domain` and `cd` into `clinicalNoteTagger/data`. You should see the newly `scp`ed `.csv` files here. Then, run `unzip newgloveicd9.txt.zip`. This generates the length-300 word vectors that the words in the notes will be converted to.

- Now it's time to actually train the model.

### Building the model

- Run `jupyter notebook --no-browser --port=8888; lsof -ti:8888 | xargs kill -9` on the server. This will boot up the instance on port 8888 on the server and instructs the server to clear the port for reuse after ending the notebook session.

- Run `ssh -N -f -L localhost:8889:localhost:8888 ssh-login@ssh_ip` on your local machine (use `-i  path/to/.pem/file` if necessary)

- In a browser tab, enter address `https://localhost:8889`. This should open up a jupyter notebook instance with all the files in the directory.

- When finishing manipulation in the jupyter notebook and stopping listening on `localhost:8889`, run `lsof -ti:8889 | xargs kill -9` on your local machine to clear port 8889.

- Finally, execute the cells in the notebook [`mimic.ipynb`](https://github.com/rivas-lab/clinicalNoteTagger/blob/master/builds/mimic.ipynb) in the [`builds`](https://github.com/rivas-lab/clinicalNoteTagger/tree/master/builds) folder which will read in the data and train the model.

### Evaluating the model

At the end of the notebook, model weights are also saved for reusing later. Use [`prediction_evaluation_mimic.ipynb`](https://github.com/rivas-lab/clinicalNoteTagger/blob/master/prediction_evaluations/prediction_evaluation_mimic.ipynb) in the [`prediction_evaluations`](https://github.com/rivas-lab/clinicalNoteTagger/tree/master/prediction_evaluations) folder to evaluate trained predictions by the Clinical Note Tagger.

There is also a [notebook](https://github.com/rivas-lab/clinicalNoteTagger/blob/master/builds/DT_RF_classification.ipynb) that shows how we trained relevant baseline classifiers (Decision Trees [DTs] and Random Forests [RFs]) and performed a hyperparameter search to determine the best performance.

## Other Datasets

FasTag can handle other datasets besides that of MIMIC, as long as your dataset of choice has clinical notes in one column and labels (hyphen-separated if multiple for one row, e.g. "1-2" or "cat:1-cat:2") in another. The numbers associated with these columns are inputted in the third coding cell in [`builds`](https://github.com/rivas-lab/clinicalNoteTagger/tree/master/builds) notebooks, e.g. [`mimic.ipynb`](https://github.com/rivas-lab/clinicalNoteTagger/blob/master/builds/mimic.ipynb). You can observe the similarities differences between [`mimic.ipynb`](https://github.com/rivas-lab/clinicalNoteTagger/blob/master/builds/mimic.ipynb) and [`csu.ipynb`](https://github.com/rivas-lab/clinicalNoteTagger/blob/master/builds/csu.ipynb), for example, in order to see how general the procedure is for building models, and can plug-and-play your dataset as needed.

[`prediction_evaluations`](https://github.com/rivas-lab/clinicalNoteTagger/tree/master/prediction_evaluations) notebooks can be run for datasets that are validated on their respective data, and [`crosschecks`](https://github.com/rivas-lab/clinicalNoteTagger/tree/master/crosschecks) notebooks can be run for evaluating models that you have built using [`builds`](https://github.com/rivas-lab/clinicalNoteTagger/tree/master/builds) notebooks but wish to test on other datasets (but, you must ensure that the labels are consistent).


##  Development status
The FasTag clinical note tagger was created in the Rivas Lab at Stanford University by Oliver Bear Don't Walk IV, Sandeep Ayyar, and Manuel Rivas for a class project in CS224N in the Winter of 2017. It was extended and continues to develop under Guhan Venkataraman in the same lab.

### Active developers
* Guhan Venkataraman (guhan[at]stanford[dot]edu)

