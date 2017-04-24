import functools
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data



def LSTM(x, weight, bias, trueWordIdxs, outputKeepProb, inputKeepProb, n_hidden, num_layers, batch_size, max_length, chatty = False):
    cell = tf.contrib.rnn.BasicLSTMCell(n_hidden,state_is_tuple = True)
    cell = tf.contrib.rnn.DropoutWrapper(cell=cell, output_keep_prob = outputKeepProb, 
                                         input_keep_prob = inputKeepProb)
    cell = tf.contrib.rnn.MultiRNNCell(cells=[cell] * num_layers, state_is_tuple=True)
#     If we ever wanna get fancy we can try the above.
    if chatty:
        print(type(cell))
        print(cell)
        print(type(x))
        print(x.get_shape())
        print('cell output size')
        print(cell.output_size)
        print('cell state size')
        print(cell.state_size)

    output, state = tf.nn.dynamic_rnn(cell = cell, inputs = x, dtype = tf.float32)
    if chatty:
        print('output shape')
        print(output.get_shape())
    # new code
    offset = tf.expand_dims(tf.range(0, batch_size, dtype = tf.int32)*max_length, 1)
    offset = tf.expand_dims(tf.range(0, tf.shape(x)[0], dtype = tf.int32)*max_length, 1)
    if chatty:
        print('offset shape')
        print(offset.get_shape())
    output = tf.reshape(output,[-1, n_hidden]) # collapses the 3d matrix into a 2d
    # matrix where all matrices are stacked on top of eachother
    if chatty:
        print('output shape new shape')
        print(output.get_shape())
    flattened_indices = trueWordIdxs + offset
    if chatty:
        print('flattened indices shape')
        print(flattened_indices.get_shape())
    output_flattened = tf.gather(output, flattened_indices)
    output_flattened = tf.reshape(output_flattened, [-1, n_hidden])
    if chatty:
        print('output flattened shape')
        print(output_flattened.get_shape())
    output_logits = tf.add(tf.matmul(output_flattened,weight),bias)
    if chatty:
        print('output wx + b')
        print(output_logits.get_shape())
    return output_logits


def doublewrap(function):
    """
    A decorator decorator, allowing to use the decorator to be used without
    parentheses if not arguments are provided. All arguments must be optional.
    """
    @functools.wraps(function)
    def decorator(*args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            return function(args[0])
        else:
            return lambda wrapee: function(wrapee, *args, **kwargs)
    return decorator


@doublewrap
def define_scope(function, scope=None, *args, **kwargs):
    """
    A decorator for functions that define TensorFlow operations. The wrapped
    function will only be executed once. Subsequent calls to it will directly
    return the result so that operations are added to the graph only once.
    The operations added by the function live within a tf.variable_scope(). If
    this decorator is used with arguments, they will be forwarded to the
    variable scope. The scope name defaults to the name of the wrapped
    function.
    """
    attribute = '_cache_' + function.__name__
    name = scope or function.__name__
    @property
    @functools.wraps(function)
    def decorator(self):
        # print('como estas bitches')
        if not hasattr(self, attribute):
            # print('1')
            with tf.variable_scope(name, *args, **kwargs):
                setattr(self, attribute, function(self))
        # print('2')
        return getattr(self, attribute)
    return decorator


class Model:

    # def __init__(self, xPlaceHolder, yPlaceHolder, embeddings, hyperParamDict):
    def __init__(self, nColsInput, nLabels, embeddings, hyperParamDict, chatty = False):
        """
        This is a doc string
        """
        # x = tf.placeholder(tf.int32, shape= (None, helper.max_length))
        # yTruth = tf.placeholder(tf.int32, shape = (None, helper.n_labels))
        # y_steps = tf.placeholder(tf.int32, shape = (None, helper.n_labels))# not sure what this is
        # trueWordIdxs = tf.placeholder(tf.int32, shape = (None,1))# vector which holds true word
        self.chatty = chatty
        self.outputKeepProb = tf.placeholder(tf.float32, shape=(), name = 'outputKeepProb')
        self.inputKeepProb = tf.placeholder(tf.float32, shape=(), name = 'inputKeepProb')
        self.maxLength = nColsInput



        self.xPlaceHolder = tf.placeholder(tf.int32, shape= (None, nColsInput), name = 'xPlaceHolder')
        self.yPlaceHolder = tf.placeholder(tf.int32, shape = (None, nLabels), name = 'yPlaceHolder')
        self.trueWordIdxs = tf.placeholder(tf.int32, shape = (None,1), name = 'trueWordIdxs')
#         self.embeddings = embeddings
        self.pretrainedEmbeddings = tf.Variable(embeddings)
        self.hyperParamDict = hyperParamDict
        self.y_last
        self.loss_function
        self.optimize
        self.alreadyLoaded = False

    @define_scope#(initializer=tf.contrib.slim.xavier_initializer())
    def y_last(self):
#         print(wtfStr)
        n_classes = int(self.yPlaceHolder.shape[1])
        x = self.xPlaceHolder
        U = tf.get_variable(name = 'U', 
                            shape = (self.hyperParamDict['n_hidden'], n_classes), 
                        initializer = tf.contrib.layers.xavier_initializer())
        bias = tf.get_variable(name = 'bias', shape = [n_classes], 
                               initializer = tf.constant_initializer(0))
#         pretrainedEmbeddings = tf.Variable(self.embeddings)
        wordEmbeddings = tf.nn.embedding_lookup(params = self.pretrainedEmbeddings, ids = x)
        if self.chatty:
            # print(self.chatty)
            print(wordEmbeddings.get_shape())
            print('shape of embeddings')
            print(wordEmbeddings.get_shape())
            print('U shape')
            print(U.get_shape())
            print('bias shape')
            print(bias.get_shape())
        y_last = LSTM(wordEmbeddings,U,bias, self.trueWordIdxs, self.outputKeepProb, self.inputKeepProb, 
            n_hidden = self.hyperParamDict['n_hidden'], num_layers = self.hyperParamDict['numLayers'],
            batch_size = self.hyperParamDict['batchSize'], max_length = self.maxLength, chatty = self.chatty)# TODO is y_last the correct thing to return?
        if self.chatty:
            print(y_last.get_shape())
        # print('como estas bitches')
        return(y_last)

    @define_scope
    def optimize(self):
        optimizer = tf.train.AdamOptimizer(
            learning_rate = self.hyperParamDict['learningRate'])
        # optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate) dis from hw3
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss_function, tvars), 
                                          self.hyperParamDict['maxGradNorm'])
        gradientVars = zip(grads, tvars)
        train_op = optimizer.apply_gradients(gradientVars)
        return(train_op)
#         logprob = tf.log(self.prediction + 1e-12)
#         cross_entropy = -tf.reduce_sum(self.label * logprob)
#         optimizer = tf.train.RMSPropOptimizer(0.03)
#         return optimizer.minimize(cross_entropy)

    @define_scope
    def loss_function(self):
        batchError = tf.nn.sigmoid_cross_entropy_with_logits(logits = self.y_last, 
                                                 labels = tf.cast(self.yPlaceHolder, tf.float32))
        loss_function = tf.reduce_mean(batchError)
        return(loss_function)

    @y_last.setter
    def y_last(self, value):
        print('checking')
        print(self.alreadyLoaded)
        if not self.alreadyLoaded:
            print('setting')
            print(value)
            self.alreadyLoaded = True
            self.y_last = 'hello world'
            return(value)
        else:
            self.alreadyLoaded = False
            print('loaded it breh')
    
    def save(self, savePath, session):
        """
        Saves the computation map so that it can be loaded later for use
        in computation, or training.
        Might remove stuff needed for training (x, trueWordsIdx)
        """
        all_saver = tf.train.Saver()
        tf.add_to_collection('y_last', self.y_last)
        tf.add_to_collection('xPlaceHolder', self.xPlaceHolder)# this just makes loading easier. Will have to
        # stop doing this in future because not necessary
        tf.add_to_collection('trueWordIdxs', self.trueWordIdxs)
        tf.add_to_collection('outputKeepProb', self.outputKeepProb)
        tf.add_to_collection('inputKeepProb', self.inputKeepProb)
        tf.add_to_collection('pretrainedEmbeddings', self.pretrainedEmbeddings)
        # tf.add_to_collection('yPlaceHolder', self.yPlaceHolder)
        all_saver.save(session, savePath)
        
        
#     def loadWeights(self, session, saverMetaPath, saverCheckPointPath):
# #         saveMetaPath = 'results/temp/bestModel.meta'
# #         saverCheckPointPath = 'results/temp/'
#         new_saver = tf.train.import_meta_graph(saverMetaPath)
#         new_saver.restore(session, tf.train.latest_checkpoint(saverCheckPointPath))
#         pretrainedEmbeddings = tf.get_collection('pretrainedEmbeddings')[0]
#         self.pretrainedEmbeddings = tf.Variable(pretrainedEmbeddings)
#         # print(tf.get_collection('y_last')[0])
#         self.y_last = tf.get_collection('y_last')[0]
#         # y_lastLoaded = tf.get_collection('y_last')[0]
#         # self.y_last.setter(value = y_lastLoaded)
#         # self.hyperParamDict = hyperParamDict
#         # return(0)


def reloadModel(session, saverMetaPath, saverCheckPointPath):
    new_saver = tf.train.import_meta_graph(saverMetaPath)
    new_saver.restore(session, tf.train.latest_checkpoint(saverCheckPointPath))
    y_last = tf.get_collection('y_last')[0]
    outputKeepProb = tf.get_collection('outputKeepProb')[0]
    inputKeepProb = tf.get_collection('inputKeepProb')[0]
    xPlaceHolder = tf.get_collection('xPlaceHolder')[0]
    trueWordIdxs = tf.get_collection('trueWordIdxs')[0]
#     y_last = tf.get_collection('y_last')[0]
    pretrainedEmbeddings = tf.get_collection('pretrainedEmbeddings')[0]
    return({'y_last':y_last,
        'outputKeepProb':outputKeepProb,
        'inputKeepProb':inputKeepProb,
        'xPlaceHolder':xPlaceHolder,
        'trueWordIdxs':trueWordIdxs,
        'pretrainedEmbeddingsm':pretrainedEmbeddings})