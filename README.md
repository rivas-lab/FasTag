# The Clinical Note Tagger
The Clinical Note Tagger was created by Oliver Bear Don't walk IV, Sandeep Ayyar, and Manuel Rivas for a class project in CS224N in the Winter of 2017. Please refer to [this paper](https://web.stanford.edu/class/cs224n/reports/2744196.pdf) for information on the Clinical Note Tagger was implemented.

# Setup TensorFlow-GPU VirtualEnv

Make sure that you have a GPU with at least 50GB RAM for training this neural network. Otherwise, the computations become intractable.

We have used Azure NV6 and Amazon P2 in the past.

First time setting up instance:

- `ssh` into the server and run `git clone https://github.com/rivas-lab/clinicalNoteTagger.git`. This should clone the repository into the instance.

- Upgrade pip for your User portal on the server: `pip install --upgrade pip --user`, 

- Install VirtualEnv: `pip install --upgrade virtualenv --user`, 

- `pip install urllib3`

- Create a VirtualEnv environment for TensorFlow: 

`virtualenv -p /usr/bin/python2.7 cnt`

- run echo $0 and find out if bash or tcsh

- Activate Environment via `source ~/cnt/bin/activate.csh` (for tcsh) or `source ~/cnt/bin/activate` (for bash)

- Install tensorflow `pip install --upgrade tensorflow-gpu`

- To come out of virtual Environment: `deactivate`

Any other time

- `ssh` into the server

- run the appropriate environment activation, `source ~/cnt/bin/activate.csh` (for tcsh) or `source ~/cnt/bin/activate` (for bash)

- To come out of virtual Environment: `deactivate`

# Run Preprocessing

- Download the script `createAdmissionNoteTable.R` from `src/textPreprocessing` to your local machine.

- Gain access to the MIMIC-III database, and then [download the files labeled DIAGNOSES_ICD.csv and NOTEEVENTS.csv here](https://physionet.org/works/MIMICIIIClinicalDatabase/files/).

- Replace the absolute paths to these two `.csv`s in lines 25 and 30 in the R script, and run the script. You should end up with three files: `icd9NotesDataTable_train.csv`, `icd9NotesDataTable_valid.csv`, and `icd9NotesDataTable.csv`.

- `scp` these three files to the server by running these commands:

`scp path/to/icd9NotesDataTable_train.csv username@domain:~/clinicalNoteTagger/data/`
`scp path/to/icd9NotesDataTable_valid.csv username@domain:~/clinicalNoteTagger/data/`
`scp path/to/icd9NotesDataTable.csv username@domain:~/clinicalNoteTagger/data/`

- In a new terminal window, `ssh username@domain` and `cd` into `clinicalNoteTagger/data`. You should see the newly `scp`ed `.csv` files here. Then, run `unzip newgloveicd9.txt.zip`.

- Now it's time to actually train the model. 

# To Train

- Run `jupyter notebook --no-browser --port=8888; lsof -ti:8888 | xargs kill -9` on the server. This will boot up the instance on port 8888 on the server and instructs the server to clear the port for reuse after ending the notebook session.

- Run `ssh -N -f -L localhost:8889:localhost:8888 ssh-login@ssh_ip` on your local machine

- In a browser tab, enter address `https://localhost:8889`. This should open up a jupyter notebook instance with all the files in the directory.

- When finishing manipulation in the jupyter notebook and stopping listening on `localhost:8889`, run `lsof -ti:8889 | xargs kill -9` on your local machine to clear port 8889.

- Finally, execute the cells in the notebook [notetaggerBuild.ipynb](https://github.com/rivas-lab/clinicalNoteTagger/blob/master/noteTaggerBuild.ipynb) which will read in the data, train the model, and run validation code.

NOTE: Fix the ipynb to run the real model and not load off the features.pkl, #dumping with 2 because ALTUD uses python 2.7 right now.

Model weights are also saved for reusing later. Please use [predictionEvaluation.ipynb](https://github.com/rivas-lab/clinicalNoteTagger/blob/master/predictionEvaluation.ipynb) to evaluate trained predictions by the Clinical Note Tagger
