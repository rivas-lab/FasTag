#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility functions to process data.
"""
import os
import pickle
import logging
from collections import Counter
import pprint

import numpy as np
import sys
sys.path.append('src/taggerSystem/')
from my_util import read_clinicalNote, one_hot, window_iterator, ConfusionMatrix, load_word_vector_mapping
from defs import LBLS, NONE, LMAP, NUM, UNK, EMBED_SIZE
import time

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)


FDIM = 4
P_CASE = "CASE:"
CASES = ["aa", "AA", "Aa", "aA"]
START_TOKEN = "<s>"
END_TOKEN = "</s>"
MAXNLABELS = 3
ICDCODELIST = []
ICDCODEDICT = {}# this allows us to map codes to integer values.

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

def normalize(word):
    """
    Normalize words that are numbers or have casing.
    """
    if word.isdigit(): return NUM
    else: return word.lower()

def featurize(embeddings, word):
    """
    Featurize a word given embeddings.
    """
    case = casing(word)
    word = normalize(word)
    case_mapping = {c: one_hot(FDIM, i) for i, c in enumerate(CASES)}
    wv = embeddings.get(word, embeddings[UNK])
    fv = case_mapping[case]
    return np.hstack((wv, fv))

def evaluate(model, X, Y):
    cm = ConfusionMatrix(labels=LBLS)
    Y_ = model.predict(X)
    for i in range(Y.shape[0]):
        y, y_ = np.argmax(Y[i]), np.argmax(Y_[i])
        cm.update(y,y_)
    cm.print_table()
    return cm.summary()

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

    def vectorize_example(self, sentence, labels=None):
        # global ICDCODEDICT
        # ICDCODEDICT = 
        sentence_ = [[self.tok2id.get(normalize(word), self.tok2id[UNK]), self.tok2id[P_CASE + casing(word)]] for word in sentence]
        if labels:
            # print('old labels')
            # print(labels)
            labels_ = np.zeros(self.n_labels)
            labels_[[ICDCODEDICT[l] for l in labels]] = 1#turning labels_ into binary vector where
            # print('new labels')
            # print(labels_)
            # print('*************************')
            # print('')
            # print('*************************')
            # 1 represents presence of disease wtih that value in ICDCODEDICT
            # if len(labels_) >= self.max_n_labels:
            #     labels_ = labels_[:self.max_n_labels]
            # else:
                # labels_.extend([None]*(self.max_n_labels - len(labels_)))
            return sentence_, labels_
        else:
            return sentence_, np.zeros()*self.n_labels
    # converts a sentence over to it's word ID and converts the case (aa (all lower). AA (all upper), aA, or Aa)
    # over to integers. Then each word becomes two features. It's word ID and the ID of the case.
    # classes are also converted into numbers. We'll have to convert out ICD9 codes to ints, or maybe we
    # can just use them as is since most of them are just numbers anyway
    def vectorize(self, data):
        return [self.vectorize_example(sentence, labels) for sentence, labels in data]


    # data should be returned from read_conll. 
    # returns a token 2 id number mapping (numbers seem to be arb), and the max length of an input
    @classmethod
    def build(cls, data):
        # Preprocess data to construct an embedding
        # Reserve 0 for the special NIL token.
        tok2id = build_dict((normalize(word) for sentence, _ in data for word in sentence), offset=1, max_words=10000) # words are put
        # to lowercase here
        # builds dict mapping words to their ID which seems to be just ordered by count and then ordered by when they were
        # added. Should be that all the words in data come before the later adds at least for value ordering anyway.
        # Not sure what this ording has to do with anything but I think everything should be ok when we have to
        # get word:embedding
        tok2id.update(build_dict([P_CASE + c for c in CASES], offset=len(tok2id)))
        tok2id.update(build_dict([START_TOKEN, END_TOKEN, UNK], offset=len(tok2id)))
        assert sorted(tok2id.items(), key=lambda t: t[1])[0][1] == 1
        logger.info("Built dictionary for %d features.", len(tok2id))

        max_length = max(len(sentence) for sentence, _ in data)
        n_labels = len(ICDCODEDICT.values())
        # print('printing token 2 id stuff')
        # print(tok2id)
        # print('')
        # print('')
        return cls(tok2id, max_length, n_labels)

    def save(self, path):
        # Make sure the directory exists.
        if not os.path.exists(path):
            os.makedirs(path)
        # Save the tok2id map.
        with open(os.path.join(path, "features.pkl"), "wb") as f:
            # print(os.path.join(path, "features.pkl"))
            # print(f)
            # print('writing this stuff')
            # print(self.tok2id)
            # print(self.max_length)
            # 1/0
            pickle.dump([self.tok2id, self.max_length], f)
        # 1/0

    @classmethod
    def load(cls, path):
        # Make sure the directory exists.
        assert os.path.exists(path) and os.path.exists(os.path.join(path, "features.pkl"))
        # Save the tok2id map.
        with open(os.path.join(path, "features.pkl")) as f:
            tok2id, max_length = pickle.load(f)
        return cls(tok2id, max_length)

def load_and_preprocess_data(data_train, data_valid):
    global ICDCODELIST
    global ICDCODEDICT
    start = time.time()
    logger.info("Loading training data...")
    train, ICDCODELIST = read_clinicalNote(data_train, icdCodeList = ICDCODELIST)
    logger.info("Done. Read %d notes", len(train))
    logger.info("Loading dev data...")
    dev, ICDCODELIST = read_clinicalNote(data_valid, ICDCODELIST)
    logger.info("Done. Read %d notes", len(dev))
    logger.info("Total read time %f", time.time() - start)
    ICDCODEDICT = {code: i for i, (code, _) in enumerate(Counter(ICDCODELIST).most_common())}
    assert len(ICDCODEDICT.values()) == len(set(ICDCODELIST))#just making sure all values are unique
    helper = ModelHelper.build(train)
    logger.info("There are a total of %d ICD codes", len(ICDCODEDICT.values()))
    # print('icd dictionary')
    # print(ICDCODEDICT)
    # now process all the input data.
    train_data = helper.vectorize(train)
    # print(train_data)
    dev_data = helper.vectorize(dev)
    print('here is the dictionary')
    print(ICDCODEDICT)
    # 1/0
    print('development data')
    print('')
    print('')
    print('')
    print('')
    print('')
    print(dev_data)
    print(dev)
    print('')
    print('')
    print('')
    print('')
    # so from what I can undersand train and dev are the raw files loaded in.
    # They can really be anything but I think tuples of [doc tokens], [doc labels]
    # will be sufficient. Remember they won't be same size since we're not tagging
    # each token.
    #train_data and dev_data are the numeric representations of these. Each word is
    # turned into two features, word ID and the upper or lower case case (aa, AA, Aa, aA)
    # We can just use word for not I think that'd be best.
    # print(helper.max_n_labels)
    # 1/0
    # 1/0
    return helper, train_data, dev_data, train, dev

# embeddings are read in from wordvecter.txt where each line correpsonds to the word embedding
# of the word on the same line in vocab.txt.
# embeddings in the end is a list of lsits where each inner list is the word embedding of a word.
# these can be accessed by tok2id[word] which returns the index into embeddings where the
# corresponding word vecter exists.
def load_embeddings(vocabPath, wordVecPath, helper):
    vocabStream = open(vocabPath, 'r')
    wordVecStream = open(wordVecPath, 'r')
    embeddings = np.array(np.random.randn(len(helper.tok2id) + 1, EMBED_SIZE), dtype=np.float32)
    # print('tokens')
    # print(helper.tok2id)
    embeddings[0] = 0.
    for word, vec in load_word_vector_mapping(vocabStream, wordVecStream).items():
        word = normalize(word)
        if word in helper.tok2id:
            # print(word)
            embeddings[helper.tok2id[word]] = vec
    logger.info("Initialized embeddings.")
    # pp = pprint.PrettyPrinter(indent=4)
    # pp.pprint(embeddings)
    vocabStream.close()
    wordVecStream.close()
    return embeddings


# i think this builds a dictionary which maps words
# to arbitrary postions. So like {I:0, you:1} etc.
# Not usre if they handle lower and uppercase here.
def build_dict(words, max_words=None, offset=0):
    cnt = Counter(words)
    if max_words:
        words = cnt.most_common(max_words)# returns most common words. Ties are arbitrarily broken
    else:
        words = cnt.most_common()
    return {word: offset+i for i, (word, _) in enumerate(words)}

def get_chunks(seq, default=LBLS.index(NONE)):
    """Breaks input of 4 4 4 0 0 4 0 ->   (0, 4, 5), (0, 6, 7)"""
    chunks = []
    chunk_type, chunk_start = None, None
    for i, tok in enumerate(seq):
        # End of a chunk 1
        if tok == default and chunk_type is not None:
            # Add a chunk.
            chunk = (chunk_type, chunk_start, i)
            chunks.append(chunk)
            chunk_type, chunk_start = None, None
        # End of a chunk + start of a chunk!
        elif tok != default:
            if chunk_type is None:
                chunk_type, chunk_start = tok, i
            elif tok != chunk_type:
                chunk = (chunk_type, chunk_start, i)
                chunks.append(chunk)
                chunk_type, chunk_start = tok, i
        else:
            pass
    # end condition
    if chunk_type is not None:
        chunk = (chunk_type, chunk_start, len(seq))
        chunks.append(chunk)
    return chunks

def test_get_chunks():
    assert get_chunks([4, 4, 4, 0, 0, 4, 1, 2, 4, 3], 4) == [(0,3,5), (1, 6, 7), (2, 7, 8), (3,9,10)]
