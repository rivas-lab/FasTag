# My data util functions

import os
import pickle
import logging
from collections import Counter
import pprint

import numpy as np
import sys
sys.path.append('src/taggerSystem/')
from util import read_clinicalNote, load_word_vector_mapping
from defs import LBLS, NONE, LMAP, NUM, UNK
import time

FDIM = 4
P_CASE = "CASE:"
CASES = ["aa", "AA", "Aa", "aA"]
START_TOKEN = "<s>"
END_TOKEN = "</s>"
MAXNLABELS = 3
ICDCODELIST = []
ICDCODEDICT = {}# this allows us to map codes to integer values.
# MAXNOTESLENGTH = 1500 # this is deprecated
def load_and_preprocess_data(data_train, data_valid, maxAllowedNoteLength, codeIdx, textIdx,
                            helperLoadPath = None):
    global ICDCODELIST
    global ICDCODEDICT
    start = time.time()
#     logger.info("Loading training data...")
    trainRaw, ICDCODELIST = read_clinicalNote(data_train, codeIdx, textIdx, icdCodeList = ICDCODELIST)
#     logger.info("Done. Read %d notes", len(trainRaw))
#     logger.info("Loading dev data...")
    devRaw, ICDCODELIST = read_clinicalNote(data_valid, codeIdx, textIdx, icdCodeList = ICDCODELIST)
#     logger.info("Done. Read %d notes", len(devRaw))
#     logger.info("Total read time %f", time.time() - start)

    ICDCODEDICT = {code: i for i, (code, _) in enumerate(Counter(ICDCODELIST).most_common())}
    assert len(ICDCODEDICT.values()) == len(set(ICDCODELIST))#just making sure all values are unique
    helper = ModelHelper.build(trainRaw, maxAllowedNoteLength, 
                               helperLoadPath = helperLoadPath)
#     if previousHelperData != None
#     ICDCODEDICT = {code: i for i, (code, _) in enumerate(Counter(ICDCODELIST).most_common())}
#     assert len(ICDCODEDICT.values()) == len(set(ICDCODELIST))#just making sure all values are unique
#     logger.info("There are a total of %d ICD codes", len(ICDCODEDICT.values()))
#     print(trainRaw)
    train_data = helper.vectorize(trainRaw)
    # print(train_data)
    dev_data = helper.vectorize(devRaw)
    xTrain, yTrain = helper.matrixify(train_data)
    xDev, yDev = helper.matrixify(dev_data)
    helper.icdDict = ICDCODEDICT
    return helper, train_data, dev_data, trainRaw, devRaw, xTrain, yTrain, xDev, yDev

# i think this builds a dictionary which maps words
# to arbitrary postions. So like {I:0, you:1} etc.
# Not sure if they handle lower and uppercase here.
def build_dict(words, max_words=None, offset=0):
    cnt = Counter(words)
    if max_words:
        words = cnt.most_common(max_words)# returns most common words. Ties are arbitrarily broken
    else:
        words = cnt.most_common()
    return {word: offset+i for i, (word, _) in enumerate(words)}

# everything is reduced to lower so might be losing information.
# all numbers map to one vector so def losing info there
def normalize(word):
    """
    Normalize words that are numbers or have casing.
    """
    if word.isdigit(): return NUM
    else: return word.lower()

# herei s where we keep information about case.... could use this again in the model
def casing(word):
    if len(word) == 0: return word

    # all lowercase
    if word.islower(): return "aa"
    # all uppercase
    elif word.isupper(): return "AA"
    # starts with capital
    elif word[0].isupper(): return "Aa"
    # has non-initial capital
    else: return "aA"
# embeddings are read in from wordvecter.txt where each line correpsonds to the word embedding
# of the word on the same line in vocab.txt.
# embeddings in the end is a list of lsits where each inner list is the word embedding of a word.
# these can be accessed by tok2id[word] which returns the index into embeddings where the
# corresponding word vecter exists.
def load_embeddings(vocabPath, wordVecPath, helper, embeddingSize):
    vocabStream = open(vocabPath, 'r')
    wordVecStream = open(wordVecPath, 'r')
    # embeddings = np.array(np.random.randn(len(helper.tok2id) + 1, EMBED_SIZE), dtype=np.float32)
    embeddings = np.array(np.random.randn(len(helper.tok2id) + 1, embeddingSize), dtype=np.float32)

    embeddings[0] = 0.
    for word, vec in load_word_vector_mapping(vocabStream, wordVecStream).items():
        word = normalize(word)
        if word in helper.tok2id:
#             print('embeddingStuff')
#             print(word)
            embeddings[helper.tok2id[word]] = vec
#     logger.info("Initialized embeddings.")
    vocabStream.close()
    wordVecStream.close()
    return embeddings



##################################Input##################################
# dataList: Object returned by vectorize. List of tuples where first elem
#   in tuple is a lit of lists [wordIdx, other features....] and the
#   second element is a binary np.array of indicating which diseases this
#   this admission was diagnosed with 
##################################Output#################################
# lastTrueWordIdxsVec: vector which has the last true word index of each
#   sentence from dataList
##################################Definition#############################
# Simply iterates through the sentences and returns the length of the
#   sentence list which corresponds to the index of the last word.
def lastTrueWordIdxs(dataList):
    sentence = []
    lastWordIdxList = np.zeros(shape = (len(dataList), 1))
    for idx, noteList in enumerate(dataList):
        sentence = noteList[0]
        # print(sentence)
        lastWordIdxList[idx] = len(sentence)-1
    # print(lastWordIdxList)
    return(lastWordIdxList)
class ModelHelper(object):
    """
    This helper takes care of preprocessing data, constructing embeddings, etc.
    """
    def __init__(self, tok2id, max_length, n_labels):
        self.tok2id = tok2id# token to arb id mapping
        self.START = [tok2id[START_TOKEN], tok2id[P_CASE + "aa"]]
        self.END = [tok2id[END_TOKEN], tok2id[P_CASE + "aa"]]
        self.max_length = max_length # max lengths of longest input
        # self.max_n_labels = max_n_labels# max number of labels for a note
        self.n_labels = n_labels# number of ICD codes in the dataset
        self.icdDict = None


    # converts a sentence over to it's word ID and converts the case (aa (all lower). AA (all upper), aA, or Aa)
    # over to integers. Then each word becomes two features. It's word ID and the ID of the case.
    # classes are also converted into numbers.
    # only takes the first max_length words in the sentence
    def vectorize_example(self, sentence, labels=None):
        sentence_ = [[self.tok2id.get(normalize(word), self.tok2id[UNK]), self.tok2id[P_CASE + casing(word)]] for word in sentence[:self.max_length]]
        if labels != ['']:
            labels_ = np.zeros(self.n_labels)
            labels_[[ICDCODEDICT[l] for l in labels]] = 1#turning labels_ into binary vector
            return sentence_, labels_
        else:
            return sentence_, np.zeros(shape = self.n_labels)
    def vectorize(self, data):
        return [self.vectorize_example(sentence, labels) for sentence, labels in data]


    # data should be returned from read_conll. 
    # returns a token 2 id number mapping (numbers seem to be arb), and the max length of an input
    @classmethod
    def build(cls, data, maxAllowedNoteLength, helperLoadPath = None):
        # Preprocess data to construct an embedding
        # Reserve 0 for the special NIL token.
        global ICDCODEDICT
        tok2id = build_dict((normalize(word) for sentence, _ in data for word in sentence), offset=1, max_words=10000)
        tok2id.update(build_dict([P_CASE + c for c in CASES], offset=len(tok2id)))
        tok2id.update(build_dict([START_TOKEN, END_TOKEN, UNK], offset=len(tok2id)))
        assert sorted(tok2id.items(), key=lambda t: t[1])[0][1] == 1
#         logger.info("Built dictionary for %d features.", len(tok2id))
        if helperLoadPath != None:
            with open(os.path.join(helperLoadPath, "features.pkl"), 'rb') as f:
                tok2id_old,maxAllowedNoteLength_old,ICDCODEDICT = pickle.load(f)
            # tok2id_old,maxAllowedNoteLength_old,ICDCODEDICT = self.load(path = helperLoadPath)
            for token in tok2id:
                if token in tok2id_old:
                    tok2id[token] = tok2id_old[token]
                else:
                    tok2id[token] = tok2id_old[UNK]
#             max_length = maxAllowedNoteLength_old
#             ICDCODEDICT = icdDictOld
        max_length = min(max(len(sentence) for sentence, _ in data), maxAllowedNoteLength)
        n_labels = len(ICDCODEDICT.values())
        icdDict = None
        return cls(tok2id, max_length, n_labels)

    def save(self, path):
        # Make sure the directory exists. 
        if not os.path.exists(path):
            os.makedirs(path)
        # Save the tok2id map.
        with open(os.path.join(path, "features.pkl"), "wb") as f:
            pickle.dump([self.tok2id, self.max_length, self.icdDict], f, protocol=2)

    @classmethod
    def load(cls, path):
        # Make sure the directory exists.
        assert os.path.exists(path) and os.path.exists(os.path.join(path, "features.pkl"))
        # Save the tok2id map.
        with open(os.path.join(path, "features.pkl"), 'rb') as f:
            tok2id, max_length, icdDict = pickle.load(f)
#         return cls(tok2id, max_length)
        return tok2id, max_length, icdDict


    ##################################Input##################################
    # dataList: Object returned by vectorize. List of tuples where first elem
    #   in tuple is a lit of lists [wordIdx, other features....] and the
    #   second element is a binary np.array of indicating which diseases this
    #   this admission was diagnosed with 
    ##################################Output#################################
    # xData: A matrix of dim (nExamples, maxLength) where entries are padded
    #   with -1. Each non-padded entry is the word ID in an example.
    # yData: A matrix where each row (obs) corresponds to the diagnosis vec
    #   for the given patient.
    ##################################Definition#############################
    # Creates a matrix X where each index corresponds to the ith word for the
    #   jth admission note. The matrix Y corresponds to the matrix of 
    #   diagnosis codes for an individual. Right now only the word index is
    #   used as a feature.
    def matrixify(self, dataList):
        """
        Creates a matrix X where each index corresponds to the ith word for the
        jth admission note. The matrix Y corresponds to the matrix of 
        diagnosis codes for an individual. Right now only the word index is
        used as a feature.
        """
        xData = np.full((len(dataList), self.max_length), -1)
        yData = np.full((len(dataList), self.n_labels), -1)
        for rowIdx, row in enumerate(dataList):
            xData[rowIdx, 0:len(row[0])] = [x[0] for x in row[0]]
            yData[rowIdx] = row[1]
        return xData, yData