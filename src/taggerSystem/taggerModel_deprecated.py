#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
"""
A model for named entity recognition.
"""
import pdb
import logging

import tensorflow as tf
from util import ConfusionMatrix, Progbar, minibatches
from data_util import get_chunks
from model import Model
from defs import LBLS

logger = logging.getLogger("hw3")
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

class taggerModel(Model):
    """
    Implements special functionality for NER models.
    """

    def __init__(self, helper, config, report=None):
        self.helper = helper
        self.config = config
        self.report = report

    def preprocess_sequence_data(self, examples):
        """Preprocess sequence data for the model.

        Args:
            examples: A list of vectorized input/output sequences.
        Returns:
            A new list of vectorized input/output pairs appropriate for the model.
        """
        raise NotImplementedError("Each Model must re-implement this method.")

    def consolidate_predictions(self, data_raw, data, preds):
        """
        Convert a sequence of predictions according to the batching
        process back into the original sequence.
        """
        raise NotImplementedError("Each Model must re-implement this method.")


    def evaluate(self, sess, examples, examples_raw):
        """Evaluates model performance on @examples.

        This function uses the model to predict labels for @examples and constructs a confusion matrix.

        Args:
            sess: the current TensorFlow session.
            examples: A list of vectorized input/output pairs.
            examples_raw: A list of the original input/output sequence pairs.
        Returns:
            The F1 score for predicting tokens as named entities.
        """
        print('here are all the values youre playing with')
        print('examples')
        print(type(examples))
        print(len(examples))
        print(examples)
        print('*************************************')
        print('*************************************')
        print('*************************************')
        print('*************************************')
        print('*************************************')
        print('*************************************')
        print('examples_raw')
        print(type(examples_raw))
        print(len(examples_raw))
        print(examples_raw)
        token_cm = ConfusionMatrix(labels=LBLS)

        correct_preds, total_correct, total_preds = 0., 0., 0.
        print('entering for loop')
        for _, labels, labels_  in self.output(sess, examples_raw, examples):
            # crashing in above call to output which gets output of model on these examples.
            # I think we need to fix preds to contrain everything as usual, but use mask to
            # index into it and get only thoese elements in  mask, and then the last pred of that
            # item. This allows us to run batch and get the prediction made at the last word
            1/0
            for l, l_ in zip(labels, labels_):
                token_cm.update(l, l_)
            print('gold')
            gold = set(get_chunks(labels))
            print('pred')
            pred = set(get_chunks(labels_))
            print(correct_preds)
            correct_preds += len(gold.intersection(pred))
            print('total preds')
            total_preds += len(pred)
            print('total corect')
            total_correct += len(gold)
        print('calculating f1 score')
        p = correct_preds / total_preds if correct_preds > 0 else 0
        r = correct_preds / total_correct if correct_preds > 0 else 0
        f1 = 2 * p * r / (p + r) if correct_preds > 0 else 0
        return token_cm, (p, r, f1)


    def run_epoch(self, sess, train_examples, dev_set, train_examples_raw, dev_set_raw):
        prog = Progbar(target=1 + int(len(train_examples) / self.config.batch_size))
        for i, batch in enumerate(minibatches(train_examples, self.config.batch_size)):
            # print(i)
            # print(batch)
            # 1/0
            loss = self.train_on_batch(sess, *batch)
            # 1/0
            prog.update(i + 1, [("train loss", loss)])
            if self.report: self.report.log_train_loss(loss)
        print("")

        #logger.info("Evaluating on training data")
        #token_cm, entity_scores = self.evaluate(sess, train_examples, train_examples_raw)
        #logger.debug("Token-level confusion matrix:\n" + token_cm.as_table())
        #logger.debug("Token-level scores:\n" + token_cm.summary())
        #logger.info("Entity level P/R/F1: %.2f/%.2f/%.2f", *entity_scores)

        logger.info("Evaluating on development data")
        # 1/0
        token_cm, entity_scores = self.evaluate(sess, dev_set, dev_set_raw)
        3/0
        logger.debug("Token-level confusion matrix:\n" + token_cm.as_table())
        logger.debug("Token-level scores:\n" + token_cm.summary())
        logger.info("Entity level P/R/F1: %.2f/%.2f/%.2f", *entity_scores)
        # 2/0
        f1 = entity_scores[-1]
        return f1

    def output(self, sess, inputs_raw, inputs=None):
        """
        Reports the output of the model on examples (uses helper to featurize each example).
        """
        if inputs is None:
            inputs = self.preprocess_sequence_data(self.helper.vectorize(inputs_raw))

        preds = []
        prog = Progbar(target=1 + int(len(inputs) / self.config.batch_size))
        for i, batch in enumerate(minibatches(inputs, self.config.batch_size, shuffle=False)):
            # Ignore predict
            batch = batch[:1] + batch[2:]
            preds_ = self.predict_on_batch(sess, *batch)
            preds += list(preds_)
            prog.update(i + 1, [])
        return self.consolidate_predictions(inputs_raw, inputs, preds)

    def fit(self, sess, saver, train_examples_raw, dev_set_raw):
        best_score = 0.

        train_examples = self.preprocess_sequence_data(train_examples_raw)
        print(type(train_examples))
        print(train_examples)
        1/0
        dev_set = self.preprocess_sequence_data(dev_set_raw)

        for epoch in range(self.config.n_epochs):
            # print('epoch stuff')
            logger.info("Epoch %d out of %d", epoch + 1, self.config.n_epochs)
            score = self.run_epoch(sess, train_examples, dev_set, train_examples_raw, dev_set_raw)
            if score > best_score:
                best_score = score
                if saver:
                    logger.info("New best score! Saving model in %s", self.config.model_output)
                    saver.save(sess, self.config.model_output)
            print("")
            if self.report:
                self.report.log_epoch()
                self.report.save()
        return best_score
