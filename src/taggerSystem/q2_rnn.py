#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Q2: Recurrent neural nets for NER
"""

from __future__ import absolute_import
from __future__ import division

import argparse
import logging
import sys
import time
from datetime import datetime

import tensorflow as tf
import numpy as np

from util import print_sentence, write_conll, read_conll
from data_util import load_and_preprocess_data, load_embeddings, ModelHelper
from taggerModel import taggerModel
from defs import LBLS
from q2_rnn_cell import RNNCell
from q3_gru_cell import GRUCell

import copy

logger = logging.getLogger("hw3.q2")
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

class Config:
    """Holds model hyperparams and data information.

    The config class is used to store various hyperparameters and dataset
    information parameters. Model objects are passed a Config() object at
    instantiation.
    """
    n_word_features = 2 # Number of features for every word in the input.
    window_size = 1
    n_features = (2 * window_size + 1) * n_word_features # Number of features for every word in the input.
    max_length = 120 # longest sequence to parse
    n_classes = 5
    max_n_labels = 100# max number of labels a note can have. Will be changed later. actually set in data_utils.py
    dropout = 0.5
    embed_size = 50
    hidden_size = 300
    batch_size = 32
    n_epochs = 10
    max_grad_norm = 10.
    lr = 0.001

    def __init__(self, args):
        self.cell = args.cell

        if "output_path" in args:
            # Where to save things.
            self.output_path = args.output_path
        else:
            self.output_path = "results/{}/{:%Y%m%d_%H%M%S}/".format(self.cell, datetime.now())
        self.model_output = self.output_path + "model.weights"
        self.eval_output = self.output_path + "results.txt"
        self.conll_output = self.output_path + "{}_predictions.conll".format(self.cell)
        self.log_output = self.output_path + "log"

def pad_sequences(data, max_length):
    """Ensures each input-output seqeunce pair in @data is of length
    @max_length by padding it with zeros and truncating the rest of the
    sequence.

    TODO: In the code below, for every sentence, labels pair in @data,
    (a) create a new sentence which appends zero feature vectors until
    the sentence is of length @max_length. If the sentence is longer
    than @max_length, simply truncate the sentence to be @max_length
    long.
    (b) create a new label sequence similarly.
    (c) create a _masking_ sequence that has a True wherever there was a
    token in the original sequence, and a False for every padded input.

    Example: for the (sentence, labels) pair: [[4,1], [6,0], [7,0]], [1,
    0, 0], and max_length = 5, we would construct
        - a new sentence: [[4,1], [6,0], [7,0], [0,0], [0,0]]
        - a new label seqeunce: [1, 0, 0, 4, 4], and
        - a masking seqeunce: [True, True, True, False, False].

    Args:
        data: is a list of (sentence, labels) tuples. @sentence is a list
            containing the words in the sentence and @label is a list of
            output labels. Each word is itself a list of
            @n_features features. For example, the sentence "Chris
            Manning is amazing" and labels "PER PER O O" would become
            ([[1,9], [2,9], [3,8], [4,8]], [1, 1, 4, 4]). Here "Chris"
            the word has been featurized as "[1, 9]", and "[1, 1, 4, 4]"
            is the list of labels. 
        max_length: the desired length for all input/output sequences.
    Returns:
        a new list of data points of the structure (sentence', labels', mask).
        Each of sentence', labels' and mask are of length @max_length.
        See the example above for more details.
    """
    ret = []

    # Use this zero vector when padding sequences.
    zero_vector = [0] * Config.n_features
    zero_label = 4 # corresponds to the 'O' tag
    for sentence, labels in data:
        ### YOUR CODE HERE (~4-6 lines)

        if len(sentence) > max_length:
            newSentence = copy.deepcopy(sentence)
            newLabels = copy.deepcopy(labels)
            ret.append((newSentence[0:max_length], newLabels, [True]*max_length))

        elif len(sentence) < max_length:
            pass
            extendLength = (max_length - len(sentence))
            mask = [True if x < len(sentence) else False for x in range(0,max_length)]
            newSentence = copy.deepcopy(sentence)
            newSentence.extend([zero_vector]*extendLength)
            newLabels = copy.deepcopy(labels)
            # newLabels.extend([zero_label]*extendLength)
            ret.append((newSentence, newLabels, mask))
        else:
            ret.append((copy.deepcopy(sentence), copy.deepcopy(labels), [True]*max_length))
            pass
        ### END YOUR CODE ###
    return ret

class RNNModel(taggerModel):
    """
    Implements a recursive neural network with an embedding layer and
    single hidden layer.
    This network will predict a sequence of labels (e.g. PER) for a
    given token (e.g. Henry) using a featurized window around the token.
    """

    def add_placeholders(self):
        """Generates placeholder variables to represent the input tensors

        These placeholders are used as inputs by the rest of the model building and will be fed
        data during training.  Note that when "None" is in a placeholder's shape, it's flexible
        (so we can use different batch sizes without rebuilding the model).

        Adds following nodes to the computational graph

        input_placeholder: Input placeholder tensor of  shape (None, self.max_length, n_features), type tf.int32
        labels_placeholder: Labels placeholder tensor of shape (None, self.max_length), type tf.int32
        mask_placeholder:  Mask placeholder tensor of shape (None, self.max_length), type tf.bool
        dropout_placeholder: Dropout value placeholder (scalar), type tf.float32

        TODO: Add these placeholders to self as the instance variables
            self.input_placeholder
            self.labels_placeholder
            self.mask_placeholder
            self.dropout_placeholder

        HINTS:
            - Remember to use self.max_length NOT Config.max_length

        (Don't change the variable names)
        """
        ### YOUR CODE HERE (~4-6 lines)
        # print('placeholder stuff')
        # print('max length')
        # print(self.max_length)
        # print('n features')
        # print(Config.n_features)
        # So I think the placeholders will have to change. instead of using max_length because we'll have 
        # a variable number of labels
        # max length of the max legnth allowable for a note
        self.input_placeholder = tf.placeholder(tf.int32, shape= (None, self.max_length, Config.n_features))
        self.labels_placeholder = tf.placeholder(tf.int32, shape= (None, Config.n_classes))
        self.mask_placeholder = tf.placeholder(tf.bool, shape= (None, self.max_length))
        self.dropout_placeholder = tf.placeholder(tf.float32, shape= ())
        # print('shape of input placeholders')
        # print(self.input_placeholder.get_shape())
        # print('shape of labels')
        # print(self.labels_placeholder.get_shape())
        # print('******************************')
        # print('')
        # print('******************************')
        # 1/0
        ### END YOUR CODE

    def create_feed_dict(self, inputs_batch, mask_batch, labels_batch=None, dropout=1):
        """Creates the feed_dict for the dependency parser.

        A feed_dict takes the form of:

        feed_dict = {
                <placeholder>: <tensor of values to be passed for placeholder>,
                ....
        }

        Hint: The keys for the feed_dict should be a subset of the placeholder
                    tensors created in add_placeholders.
        Hint: When an argument is None, don't add it to the feed_dict.

        Args:
            inputs_batch: A batch of input data.
            mask_batch:   A batch of mask data.
            labels_batch: A batch of label data.
            dropout: The dropout rate.
        Returns:
            feed_dict: The feed dictionary mapping from placeholders to values.
        """
        ### YOUR CODE (~6-10 lines)
        feed_dict = {}
        if labels_batch is None:
            feed_dict = {self.input_placeholder: inputs_batch, self.dropout_placeholder: dropout, self.mask_placeholder: mask_batch}
        else:
            feed_dict = {self.input_placeholder: inputs_batch, self.dropout_placeholder: dropout, self.mask_placeholder: mask_batch, self.labels_placeholder: labels_batch}
        ### END YOUR CODE
        return feed_dict

    def add_embedding(self):
        """Adds an embedding layer that maps from input tokens (integers) to vectors and then
        concatenates those vectors:

        TODO:
            - Create an embedding tensor and initialize it with self.pretrained_embeddings.
            - Use the input_placeholder to index into the embeddings tensor, resulting in a
              tensor of shape (None, max_length, n_features, embed_size).
            - Concatenates the embeddings by reshaping the embeddings tensor to shape
              (None, max_length, n_features * embed_size).

        HINTS:
            - You might find tf.nn.embedding_lookup useful.
            - You can use tf.reshape to concatenate the vectors. See
              following link to understand what -1 in a shape means.
              https://www.tensorflow.org/api_docs/python/array_ops/shapes_and_shaping#reshape.

        Returns:
            embeddings: tf.Tensor of shape (None, max_length, n_features*embed_size)
        """
        ### YOUR CODE HERE (~4-6 lines)
        pretrainedEmbeddings = tf.Variable(self.pretrained_embeddings)

        embeddings = tf.nn.embedding_lookup(params = pretrainedEmbeddings, ids =self.input_placeholder)
        # print('embedding shape before reshape')
        # print(embeddings.get_shape())

        # embeddings = tf.reshape(embeddings, shape = tf.pack([-1, embeddings.get_shape()[1], embeddings.get_shape()[2]*embeddings.get_shape()[3]]))
        embeddings = tf.reshape(embeddings, shape = tf.stack([-1, embeddings.get_shape()[1], embeddings.get_shape()[2]*embeddings.get_shape()[3]])) 
        # print('embedding shape after reshape')
        # print(embeddings.get_shape())
        # print('******************************')
        # print('')
        # print('******************************')
        # 1/0
        # changed above. pack -> stack because of new tf version
        ### END YOUR CODE
        return embeddings

    def add_prediction_op(self):
        """Adds the unrolled RNN:
            h_0 = 0
            for t in 1 to T:
                o_t, h_t = cell(x_t, h_{t-1})
                o_drop_t = Dropout(o_t, dropout_rate)
                y_t = o_drop_t U + b_2

        TODO: There a quite a few things you'll need to do in this function:
            - Define the variables U, b_2.
            - Define the vector h as a constant and inititalize it with
              zeros. See tf.zeros and tf.shape for information on how
              to initialize this variable to be of the right shape.
              https://www.tensorflow.org/api_docs/python/constant_op/constant_value_tensors#zeros
              https://www.tensorflow.org/api_docs/python/array_ops/shapes_and_shaping#shape
            - In a for loop, begin to unroll the RNN sequence. Collect
              the predictions in a list.
            - When unrolling the loop, from the second iteration
              onwards, you will HAVE to call
              tf.get_variable_scope().reuse_variables() so that you do
              not create new variables in the RNN cell.
              See https://www.tensorflow.org/versions/master/how_tos/variable_scope/
            - Concatenate and reshape the predictions into a predictions
              tensor.
        Hint: You will find the function tf.pack (similar to np.asarray)
              useful to assemble a list of tensors into a larger tensor.
              https://www.tensorflow.org/api_docs/python/array_ops/slicing_and_joining#pack
        Hint: You will find the function tf.transpose and the perms
              argument useful to shuffle the indices of the tensor.
              https://www.tensorflow.org/api_docs/python/array_ops/slicing_and_joining#transpose

        Remember:
            * Use the xavier initilization for matrices.
            * Note that tf.nn.dropout takes the keep probability (1 - p_drop) as an argument.
            The keep probability should be set to the value of self.dropout_placeholder

        Returns:
            pred: tf.Tensor of shape (batch_size, max_length, n_classes)
        """
        x = self.add_embedding()
        dropout_rate = self.dropout_placeholder

        preds = [] # Predicted output at each timestep should go here!

        # Use the cell defined below. For Q2, we will just be using the
        # RNNCell you defined, but for Q3, we will run this code again
        # with a GRU cell!
        if self.config.cell == "rnn":
            cell = RNNCell(Config.n_features * Config.embed_size, Config.hidden_size)
        elif self.config.cell == "gru":
            cell = GRUCell(Config.n_features * Config.embed_size, Config.hidden_size)
        else:
            raise ValueError("Unsuppported cell type: " + self.config.cell)

        # Define U and b2 as variables.
        # Initialize state as vector of zeros.
        ### YOUR CODE HERE (~4-6 lines)
        # 3/0
        with tf.variable_scope('RNN_OutsideCell', reuse = False) as scope:
            # 4/0
            # print('shape of outside cell stuff U')
            U = tf.get_variable(name = 'U', shape = (Config.hidden_size, Config.n_classes), 
                initializer = tf.contrib.layers.xavier_initializer())
            # 5/0
            # print(U.get_shape())
            b2 = tf.get_variable(name = 'b2', shape = [Config.n_classes], initializer = tf.constant_initializer(0))
            # 6/0
            # print('shape of b2')
            # print(b2.get_shape())
            state = tf.zeros(name = 'state', shape = (tf.shape(x)[0],Config.hidden_size))# can either do one or Config.batch_size
        #     print('shape of state')
        #     print(state.get_shape())
        # print('******************************')
        # print('')
        # print('******************************')
        # 2/0
        ### END YOUR CODE

        with tf.variable_scope("RNN"):
            for time_step in range(self.max_length):
                ### YOUR CODE HERE (~6-10 lines)
                if time_step == 1:#second time step
                    tf.get_variable_scope().reuse_variables()
                    # now the model knows to reuse variables... right? Idk man... i'm just here to do my job.
                ot, state = cell(x[:,time_step,:], state)
                o_drop_t = tf.nn.dropout(x = ot, keep_prob = dropout_rate)
                yHat = tf.matmul(o_drop_t, U) + b2
                # print('yhat shape')
                # print(yHat.get_shape())
                preds.append(yHat)
                # Here is where you should only append preds of last one because you don't care about 
                # preds before. Don't count htose as error.                
                # 1/0
                pass
                ### END YOUR CODE
        preds = preds[-1]# only care about the very last error
        # Make sure to reshape @preds here.
        ### YOUR CODE HERE (~2-4 lines)
        # preds = tf.pack(preds)
        preds = tf.stack(preds) # once again no pack argument
        # print('preds shape')
        # print(preds.get_shape())
        # 2/0
        # preds = tf.transpose(preds, perm = [1, 0, 2])
        ### END YOUR CODE

        # assert preds.get_shape().as_list() == [None, self.max_length, self.config.n_classes], "predictions are not of the right shape. Expected {}, got {}".format([None, self.max_length, self.config.n_classes], preds.get_shape().as_list())
        # should prolly put a warning here
        # # TODO ^
        # print('preds shape')
        # print(preds.get_shape())
        # print('******************************')
        # print('')
        # print('******************************')
        return preds

    def add_loss_op(self, preds):
        """Adds Ops for the loss function to the computational graph.

        TODO: Compute averaged cross entropy loss for the predictions.
        Importantly, you must ignore the loss for any masked tokens.

        Hint: You might find tf.boolean_mask useful to mask the losses on masked tokens.
        Hint: You can use tf.nn.sparse_softmax_cross_entropy_with_logits to simplify your
                    implementation. You might find tf.reduce_mean useful.
        Args:
            pred: A tensor of shape (batch_size, max_length, n_classes) containing the output of the neural
                  network before the softmax layer.
        Returns:
            loss: A 0-d tensor (scalar)
        """
        ### YOUR CODE HERE (~2-4 lines)
        # 1/0 
        # print('preds shape before')
        # print(preds.get_shape())
        # print('bales shape')
        # print(self.labels_placeholder.get_shape())
        # preds = 
        # print(self.labels_placeholder)
        # batchError = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = preds, labels = self.labels_placeholder)
        batchError = tf.nn.softmax_cross_entropy_with_logits(logits = preds, labels = self.labels_placeholder)
        # changed to above because I think the other one was for calculating loss at each step
        # print('batch error shape')
        # print(batchError.get_shape())
        # print('mask pkace hoolder shape')
        # print(self.mask_placeholder.get_shape())
        # 3/0
        # loss = tf.reduce_mean(tf.boolean_mask(batchError, self.mask_placeholder))
        loss = tf.reduce_mean(batchError) 
        # cahnged above because no longer need masking     
        # 2/0               
        ### END YOUR CODE
        return loss

    def add_training_op(self, loss):
        """Sets up the training Ops.

        Creates an optimizer and applies the gradients to all trainable variables.
        The Op returned by this function is what must be passed to the
        `sess.run()` call to cause the model to train. See

        https://www.tensorflow.org/versions/r0.7/api_docs/python/train.html#Optimizer

        for more information.

        Use tf.train.AdamOptimizer for this model.
        Calling optimizer.minimize() will return a train_op object.

        Args:
            loss: Loss tensor, from cross_entropy_loss.
        Returns:
            train_op: The Op for training.
        """
        ### YOUR CODE HERE (~1-2 lines)
        train_op = tf.train.AdamOptimizer(learning_rate = self.config.lr).minimize(loss)
        ### END YOUR CODE
        return train_op

    def preprocess_sequence_data(self, examples):
        def featurize_windows(data, start, end, window_size = 1):
            """Uses the input sequences in @data to construct new windowed data points.
            """
            # 1/0
            ret = []
            for sentence, labels in data:
                from util import window_iterator
                sentence_ = []
                for window in window_iterator(sentence, window_size, beg=start, end=end):
                    sentence_.append(sum(window, []))
                ret.append((sentence_, labels))
            return ret

        examples = featurize_windows(examples, self.helper.START, self.helper.END)
        return pad_sequences(examples, self.max_length)

    def consolidate_predictions(self, examples_raw, examples, preds):
        """Batch the predictions into groups of sentence length.
        """
        assert len(examples_raw) == len(examples)
        assert len(examples_raw) == len(preds)

        ret = []
        for i, (sentence, labels) in enumerate(examples_raw):
            _, _, mask = examples[i]
            labels_ = [l for l, m in zip(preds[i], mask) if m] # only select elements of mask.
            assert len(labels_) == len(labels)
            ret.append([sentence, labels, labels_])
        return ret

    def predict_on_batch(self, sess, inputs_batch, mask_batch):
        feed = self.create_feed_dict(inputs_batch=inputs_batch, mask_batch=mask_batch)
        predictions = sess.run(tf.argmax(self.pred, axis=2), feed_dict=feed)
        return predictions

    def train_on_batch(self, sess, inputs_batch, labels_batch, mask_batch):
        feed = self.create_feed_dict(inputs_batch, labels_batch=labels_batch, mask_batch=mask_batch,
                                     dropout=Config.dropout)
        _, loss = sess.run([self.train_op, self.loss], feed_dict=feed)
        return loss

    def __init__(self, helper, config, pretrained_embeddings, report=None):
        super(RNNModel, self).__init__(helper, config, report)
        self.max_length = min(Config.max_length, helper.max_length)
        Config.max_length = self.max_length # Just in case people make a mistake.
        self.pretrained_embeddings = pretrained_embeddings
        # print('first')
        # print(Config.max_n_labels)
        Config.max_n_labels = helper.max_n_labels
        # print('then')
        # print(Config.max_n_labels)

        # Defining placeholders.
        self.input_placeholder = None
        self.labels_placeholder = None
        self.mask_placeholder = None
        self.dropout_placeholder = None

        self.build()

def test_pad_sequences():
    Config.n_features = 2
    data = [
        ([[4,1], [6,0], [7,0]], [1, 0, 0]),
        ([[3,0], [3,4], [4,5], [5,3], [3,4]], [0, 1, 0, 2, 3]),
        ]
    ret = [
        ([[4,1], [6,0], [7,0], [0,0]], [1, 0, 0, 4], [True, True, True, False]),
        ([[3,0], [3,4], [4,5], [5,3]], [0, 1, 0, 2], [True, True, True, True])
        ]

    ret_ = pad_sequences(data, 4)
    assert len(ret_) == 2, "Did not process all examples: expected {} results, but got {}.".format(2, len(ret_))
    for i in range(2):
        assert len(ret_[i]) == 3, "Did not populate return values corrected: expected {} items, but got {}.".format(3, len(ret_[i]))
        for j in range(3):
            assert ret_[i][j] == ret[i][j], "Expected {}, but got {} for {}-th entry of {}-th example".format(ret[i][j], ret_[i][j], j, i)

def do_test1(_):
    logger.info("Testing pad_sequences")
    test_pad_sequences()
    logger.info("Passed!")

def do_test2(args):
    logger.info("Testing implementation of RNNModel")
    config = Config(args)
    helper, train, dev, train_raw, dev_raw = load_and_preprocess_data(args)
    embeddings = load_embeddings(args, helper)
    config.embed_size = embeddings.shape[1]

    with tf.Graph().as_default():
        logger.info("Building model...",)
        start = time.time()
        model = RNNModel(helper, config, embeddings)
        logger.info("took %.2f seconds", time.time() - start)

        init = tf.global_variables_initializer()
        saver = None

        with tf.Session() as session:
            session.run(init)
            model.fit(session, saver, train, dev)

    logger.info("Model did not crash!")
    logger.info("Passed!")

def do_train(args):
    # Set up some parameters.
    config = Config(args)
    helper, train, dev, train_raw, dev_raw = load_and_preprocess_data(args)
    embeddings = load_embeddings(args, helper)
    config.embed_size = embeddings.shape[1]
    helper.save(config.output_path)# token2id and max length saved to output_path
    handler = logging.FileHandler(config.log_output)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
    logging.getLogger().addHandler(handler)

    report = None #Report(Config.eval_output)

    with tf.Graph().as_default():
        logger.info("Building model...",)
        start = time.time()
        model = RNNModel(helper, config, embeddings)
        logger.info("took %.2f seconds", time.time() - start)

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        with tf.Session() as session:
            session.run(init)
            model.fit(session, saver, train, dev)
            if report:
                report.log_output(model.output(session, dev_raw))
                report.save()
            else:
                # Save predictions in a text file.
                output = model.output(session, dev_raw)
                sentences, labels, predictions = zip(*output)
                predictions = [[LBLS[l] for l in preds] for preds in predictions]
                output = zip(sentences, labels, predictions)

                with open(model.config.conll_output, 'w') as f:
                    write_conll(f, output)
                with open(model.config.eval_output, 'w') as f:
                    for sentence, labels, predictions in output:
                        print_sentence(f, sentence, labels, predictions)

def do_evaluate(args):
    config = Config(args.model_path)
    helper = ModelHelper.load(args.model_path)
    input_data = read_conll(args.data)
    embeddings = load_embeddings(args, helper)
    config.embed_size = embeddings.shape[1]

    with tf.Graph().as_default():
        logger.info("Building model...",)
        start = time.time()
        model = RNNModel(helper, config, embeddings)

        logger.info("took %.2f seconds", time.time() - start)

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        with tf.Session() as session:
            session.run(init)
            saver.restore(session, model.config.model_output)
            for sentence, labels, predictions in model.output(session, input_data):
                predictions = [LBLS[l] for l in predictions]
                print_sentence(args.output, sentence, labels, predictions)

def do_shell(args):
    config = Config(args.model_path)
    helper = ModelHelper.load(args.model_path)
    embeddings = load_embeddings(args, helper)
    config.embed_size = embeddings.shape[1]

    with tf.Graph().as_default():
        logger.info("Building model...",)
        start = time.time()
        model = RNNModel(helper, config, embeddings)
        logger.info("took %.2f seconds", time.time() - start)

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        with tf.Session() as session:
            session.run(init)
            saver.restore(session, model.config.model_output)

            print("""Welcome!
You can use this shell to explore the behavior of your model.
Please enter sentences with spaces between tokens, e.g.,
input> Germany 's representative to the European Union 's veterinary committee .
""")
            while True:
                # Create simple REPL
                try:
                    sentence = raw_input("input> ")
                    tokens = sentence.strip().split(" ")
                    for sentence, _, predictions in model.output(session, [(tokens, ["O"] * len(tokens))]):
                        predictions = [LBLS[l] for l in predictions]
                        print_sentence(sys.stdout, sentence, [""] * len(tokens), predictions)
                except EOFError:
                    print("Closing session.")
                    break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Trains and tests an NER model')
    subparsers = parser.add_subparsers()

    command_parser = subparsers.add_parser('test1', help='')
    # command_parser.set_defaults(func=do_test1)

    command_parser = subparsers.add_parser('test2', help='')
    # command_parser.add_argument('-dt', '--data-train', type=argparse.FileType('r'), default="data/tiny.conll", help="Training data")
    # command_parser.add_argument('-dd', '--data-dev', type=argparse.FileType('r'), default="data/tiny.conll", help="Dev data")
    # command_parser.add_argument('-v', '--vocab', type=argparse.FileType('r'), default="data/vocab.txt", help="Path to vocabulary file")
    # command_parser.add_argument('-vv', '--vectors', type=argparse.FileType('r'), default="data/wordVectors.txt", help="Path to word vectors file")
    # command_parser.add_argument('-c', '--cell', choices=["rnn", "gru"], default="rnn", help="Type of RNN cell to use.")
    # command_parser.set_defaults(func=do_test2)

    command_parser = subparsers.add_parser('train', help='')
    command_parser.add_argument('-dt', '--data-train', type=argparse.FileType('r'), default="data/icd9NotesDataTable_train.csv", help="Training data")
    command_parser.add_argument('-dd', '--data-dev', type=argparse.FileType('r'), default="data/icd9NotesDataTable_valid.csv", help="Dev data")
    command_parser.add_argument('-v', '--vocab', type=argparse.FileType('r'), default="src/taggerSystem/data_hw3_delete/vocab.txt", help="Path to vocabulary file")
    command_parser.add_argument('-vv', '--vectors', type=argparse.FileType('r'), default="src/taggerSystem/data_hw3_delete/wordVectors.txt", help="Path to word vectors file")
    command_parser.add_argument('-c', '--cell', choices=["rnn", "gru"], default="rnn", help="Type of RNN cell to use.")
    command_parser.set_defaults(func=do_train)

    command_parser = subparsers.add_parser('evaluate', help='')
    # command_parser.add_argument('-d', '--data', type=argparse.FileType('r'), default="data/dev.conll", help="Training data")
    # command_parser.add_argument('-m', '--model-path', help="Training data")
    # command_parser.add_argument('-v', '--vocab', type=argparse.FileType('r'), default="data/vocab.txt", help="Path to vocabulary file")
    # command_parser.add_argument('-vv', '--vectors', type=argparse.FileType('r'), default="data/wordVectors.txt", help="Path to word vectors file")
    # command_parser.add_argument('-c', '--cell', choices=["rnn", "gru"], default="rnn", help="Type of RNN cell to use.")
    # command_parser.add_argument('-o', '--output', type=argparse.FileType('w'), default=sys.stdout, help="Training data")
    # command_parser.set_defaults(func=do_evaluate)

    # command_parser = subparsers.add_parser('shell', help='')
    # command_parser.add_argument('-m', '--model-path', help="Training data")
    # command_parser.add_argument('-v', '--vocab', type=argparse.FileType('r'), default="data/vocab.txt", help="Path to vocabulary file")
    # command_parser.add_argument('-vv', '--vectors', type=argparse.FileType('r'), default="data/wordVectors.txt", help="Path to word vectors file")
    # command_parser.add_argument('-c', '--cell', choices=["rnn", "gru"], default="rnn", help="Type of RNN cell to use.")
    # command_parser.set_defaults(func=do_shell)

    ARGS = parser.parse_args()
    if ARGS.func is None:
        parser.print_help()
        sys.exit(1)
    else:
        ARGS.func(ARGS)
