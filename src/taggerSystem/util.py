from __future__ import division

import sys
import time
import logging
from io import StringIO
from collections import defaultdict, Counter, OrderedDict
import numpy as np
from numpy import array, zeros, allclose
import csv
def read_clinicalNote(path, codeIdx, textIdx, icdCodeList = []):
    """
    Reads in a clinical note and returns a list of tokens as well as the ICD9 codes 
    associated with the file. 
    Example: 

    Attributes:

    Args:
        path str: Path to clinical note csv
        codeIdx int: Column index which contains icd codes
        textIdx int: Column index which contains textIdx
        icdCodeList list: List which will contian all unique icd9 codes
        

    Returns:

    TODO:
        1)
    """
    expectedHeader = ['', 'HADM_ID', 'SUBJECT_ID', 'ICD9_CODE', 'CHARTDATE', 'DESCRIPTION', 'TEXT'] # old version.
    expectedHeader = ["","HADM_ID","SUBJECT_ID","ICD9_CODE","CHARTDATE","DESCRIPTION","TEXT","Level2ICD","TopLevelICD","V9"]
    # codeIdx = 9
    # textIdx = 6
    ret = []

    current_toks, current_lbls = [], []
    # print(path)
    with open(path, 'r') as csvfile:
        csvReader = csv.reader(csvfile, delimiter=',', quotechar='\"', skipinitialspace=True)
        assert next(csvReader) == expectedHeader #checking that the header matches what we expect. 
        for row in csvReader:
            # print(row[codeIdx].split('-'))
            ret.append((row[textIdx].split(), row[codeIdx].split('-')))
            icdCodeList.extend(row[codeIdx].split('-'))
    # 1/0
    return(ret, list(set(icdCodeList)))

def load_word_vector_mapping(vocab_fstream, vector_fstream):
    """
    Load word vector mapping using @vocab_fstream, @vector_fstream.
    Assumes each line of the vocab file matches with those of the vector
    file.
    Example: 

    Attributes:

    Args:
        vocab_fstream fstream: Stream to where the vocab file is
        vector_fstream fstream: stream to where the word vector file is
        

    Returns:
        ret dict: a dict which maps vocab to word vectors.
    TODO:
        1)
    """
    ret = OrderedDict()
    for vocab, vector in zip(vocab_fstream, vector_fstream):
        vocab = vocab.strip()
        vector = vector.strip()
        ret[vocab] = array(list(map(float, vector.split())))
    # print(ret['UUUNKKK'])
    # print(ret['the'])
    # 1/0
    return ret
def one_hot(n, y):
    """
    Create a one-hot @n-dimensional vector with a 1 in position @i
    """
    if isinstance(y, int):
        ret = zeros(n)
        ret[y] = 1.0
        return ret
    elif isinstance(y, list):
        ret = zeros((len(y), n))
        ret[np.arange(len(y)),y] = 1.0
        return ret
    else:
        raise ValueError("Expected an int or list got: " + y)