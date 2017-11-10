# The Clinical Note Tagger
The Clinical Note Tagger was created by Oliver Bear Don't walk IV, Sandeep Ayyar, and Manuel Rivas for a class project in CS224N in the Winter of 2017. Please refer to [this paper](https://web.stanford.edu/class/cs224n/reports/2744196.pdf) for information on the Clinical Note Tagger was implemented.

# To Train

Make sure that you have a GPU with at least 50GB RAM for training this neural network. Otherwise, the computations become intractable.

We have used Azure NV6 and Amazon P2 in the past.

ssh into the server and run `git clone https://github.com/rivas-lab/clinicalNoteTagger.git`. This should clone the repository into the instance.

Upgrade pip for your User portal on the server: `pip install --upgrade pip --user`, 

Install VirtualEnv: `pip install --upgrade virtualenv --user`, 

`pip install urllib3`

Create a VirtualEnv environment for TensorFlow: 

`virtualenv -p /usr/bin/python2.7 cnt`

run echo $0 and find out if bash or tcsh

Activate Environment via `source ~/cnt/bin/activate.csh` (for tcsh) or `source ~/cnt/bin/activate` (for bash)

Install tensorflow `pip install --upgrade tensorflow-gpu`

(To come out of virtual Environment: deactivate)

`jupyter notebook --no-browser`
on local machine, run command with `xxxx` as the port that the jupyter notebook runs at
`ssh -N -f -L localhost:8889:localhost:xxxx ssh-login@ssh_ip`

Now in a browser tab, enter address `https://localhost:8889`

Execute the cells in the notebook [notetaggerBuild.ipynb](https://github.com/rivas-lab/clinicalNoteTagger/blob/master/noteTaggerBuild.ipynb) which will read in the data, train the model, and run validation code. Model weights are also saved for reusing later. Please use [predictionEvaluation.ipynb](https://github.com/rivas-lab/clinicalNoteTagger/blob/master/predictionEvaluation.ipynb) to evaluate trained predictions by the Clinical Note Tagger
