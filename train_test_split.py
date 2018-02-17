import pandas as pd
import numpy as np

#put absolute or relative path to csv with all data here
df = pd.read_csv('final_csu_file', sep='\t')

#set the proportion of train data here
train_proportion = 0.7

msk = np.random.rand(len(df)) <= train_proportion

train = df[msk]
test = df[~msk]

#Put in the names of the train and test files you would like as arguments here
train.to_csv('csu_snomed_train', sep='\t', index=False)
test.to_csv('csu_snomed_test', sep='\t', index=False)
