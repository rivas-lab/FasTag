import functools
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data






#Note: sizeList must have the n_hidden as the first size and n_classes
# as the last size and then anything else in between is gravy
def feedForwardNNet(sizeList, lstmHiddenState, chatty = False):
    """
        Hard coded different feed forward nnets to add to the end of the LSTM.
        Not sure if this can be done automatically, but this way it's pretty easy
        to create different length nnets to tack on.
        Example: 

        Attributes:

        Args:
            sizeList (list): Gives the size of the weight and bias matrices
                that the fully connected nnets should use. The first element
                should be the size of the last hiddent state from the LSTM
                (or whatever the LSTM outputs) and the last element should be
                the number of classes.
                [200, 150, 75, 19] for example
            lstmHiddenState (tensor): Should be the vector output by the LSTM cell.
            chatty (bool): I forgot the word verbose, so this tells the function 
                whether or not it should be printing

        Returns:
            output_logits: These are the final vector of the FCNNET and still need
                to be processed to get actual predictions from them
        TODO:
            1) Would be nice to have some non-linearity.
    """
    assert len(sizeList) in [2, 4], "This size list has not been coded up yet"
    if len(sizeList) == 2:
        U = tf.get_variable(name = 'U', 
                    shape = (sizeList[0], sizeList[1]), 
                    initializer = tf.contrib.layers.xavier_initializer())
        bias = tf.get_variable(name = 'bias', shape = [sizeList[1]], 
                    initializer = tf.constant_initializer(0))
        print('hello')
        if chatty:
            print('U shape')
            print(U.get_shape())
            print('bias shape')
            print(bias.get_shape())
        output_logits = tf.add(tf.matmul(lstmHiddenState,U),bias)
    if len(sizeList) == 4:
        # 1/0
        # 1/0
        # tf.nn.dropout(h_fc1, keep_prob)
        # add dropout with above code
        W_1 = tf.get_variable(name = 'W_1', 
                            shape = (sizeList[0], sizeList[1]), 
                        initializer = tf.contrib.layers.xavier_initializer())
        b_1 = tf.get_variable(name = 'b_1', shape = [sizeList[1]], 
                               initializer = tf.constant_initializer(0))

        W_2 = tf.get_variable(name = 'W_2', 
                            shape = (sizeList[1], sizeList[2]), 
                        initializer = tf.contrib.layers.xavier_initializer())
        b_2 = tf.get_variable(name = 'b_2', shape = [sizeList[2]], 
                               initializer = tf.constant_initializer(0))

        U = tf.get_variable(name = 'U', 
                            shape = (sizeList[2], sizeList[3]), 
                        initializer = tf.contrib.layers.xavier_initializer())
        bias = tf.get_variable(name = 'bias', shape = [sizeList[3]], 
                               initializer = tf.constant_initializer(0))
        if chatty:
            print(W_1.get_shape())
            print('W_1 shape')
            print(b_1.get_shape())
            print('b_1 shape')
            print(W_2.get_shape())
            print('W_2 shape')
            print(b_2.get_shape())
            print('bias shape')
            print(bias.get_shape())
            print('U shape')
            print(U.get_shape())
            print('bias shape')
            print(bias.get_shape())
        output_logits = tf.nn.relu(tf.matmul(lstmHiddenState, W_1) + b_1)
        output_logits = tf.nn.relu(tf.matmul(output_logits, W_2) + b_2)
        output_logits = tf.matmul(output_logits, U) + bias
    
    return(output_logits)





def LSTM(x, sizeList, trueWordIdxs, outputKeepProb, inputKeepProb, n_hidden, 
    num_layers, batch_size, max_length, chatty = False):
    """
        This is the entire implementation for the LSTM and the fully connected nnet 
        attached the the last hidden state of the LSTM. Predictions are not output 
        here so there's a bit more work that needs to be done when you get them.
        They're basically logits still. 
        Example: 

        Attributes:

        Args:
            x (tf placeholder): Input for the LSTM. Shape = (None, nObs, maxWordLength)
            sizeList (list): Gives the size of the weight and bias matrices
                that the fully connected nnets should use. The first element
                should be the size of the last hiddent state from the LSTM
                (or whatever the LSTM outputs) and the last element should be
                the number of classes.
                [200, 150, 75, 19] for example
            trueWordIdxs (tf placeholder): Each index tells the model which word is the
                last true word since we had to make all notes the same length for batch
                processing there is some padding going on.
            outputKeepProb (tf placeholder): Essentially a float between 0 and 1 which is 
                the keep probability for the hidden state of each LSTM cell
            inputKeepProb (tf placeholder): Similar to above except it is for the input 
                words. Doesn't seem to make a big difference so might as well keep at 1
            n_hidden (int): Size of the hidden layer.
            num_layers (int): How many stacked LSTMs to use.
            batch_size (int): Batch size so I can go and pick out the last hidden state
                which is used for prediction
            max_length (int): Maximum length for the notes, enables batch processing. 
            chatty (bool): I forgot the word verbose, so this tells the function 
                whether or not it should be printin          

        Returns:
            output_logits (tensor): Vector with the logits for each observation. Will 
                be of size (nObs, nClasses).
        TODO:
            1) nice to have bi directional RNN
    """
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
    output_logits = feedForwardNNet(sizeList, output_flattened, chatty = chatty)
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
    def __init__(self, nColsInput, nLabels, embeddings, hyperParamDict, sizeList, chatty = False):
        """
        Creates the model by building the computationa graph. Please note that the code above
        for the decorator was taken from Danijar Hafner's blog post at 
        https://danijar.com/structuring-your-tensorflow-models/. It lets me create the model in 
        a modular fashion and then doesn't recreate it everytime one of the attributes is called.
        Example: 

        Attributes:
            y_last: The prediction part of the computationa graph. it runs the LSTM and the neural 
                network at the end.
            optimize: Runs adam optimization with the given learning rate and some gradient clipping.
            loss_function: Computes the loss using cross entropy.


        Args:
            nColsInput (int): Maximum length for the notes, enables batch processing.
            nLabels (int): How many labels are being used for prediction. The number of ICD9 codes
            embeddings (tf variable): Word embeddings.
            hyperParamDict: Dictionary for all the tunable parameters and information about the model
                being built.
            sizeList (list): Gives the size of the weight and bias matrices
                that the fully connected nnets should use. The first element
                should be the size of the last hiddent state from the LSTM
                (or whatever the LSTM outputs) and the last element should be
                the number of classes.
                [200, 150, 75, 19] for example
            chatty (bool): I forgot the word verbose, so this tells the function 
                whether or not it should be printing
            

        Returns:
            Model Class Object. Lets you encapsulate the model in one object and access it through
                attributes.
            
        TODO:
            1) Add example for the hyper parameter dict
        """
        # x = tf.placeholder(tf.int32, shape= (None, helper.max_length))
        # yTruth = tf.placeholder(tf.int32, shape = (None, helper.n_labels))
        # y_steps = tf.placeholder(tf.int32, shape = (None, helper.n_labels))# not sure what this is
        # trueWordIdxs = tf.placeholder(tf.int32, shape = (None,1))# vector which holds true word
        self.chatty = chatty
        assert (sizeList[0] == hyperParamDict['n_hidden']) and (sizeList[-1] == nLabels), \
            "Size list must start with the number of hidden units and end with the number of labels"
        self.sizeList = sizeList
        self.hyperParamDict = hyperParamDict
        self.outputKeepProb = tf.placeholder(tf.float32, shape=(), name = 'outputKeepProb')
        self.inputKeepProb = tf.placeholder(tf.float32, shape=(), name = 'inputKeepProb')
        self.maxLength = nColsInput



        self.xPlaceHolder = tf.placeholder(tf.int32, shape= (None, nColsInput), name = 'xPlaceHolder')
        self.yPlaceHolder = tf.placeholder(tf.int32, shape = (None, nLabels), name = 'yPlaceHolder')
        self.trueWordIdxs = tf.placeholder(tf.int32, shape = (None,1), name = 'trueWordIdxs')
#         self.embeddings = embeddings
        self.pretrainedEmbeddings = tf.Variable(embeddings)
        self.y_last
        self.loss_function
        self.optimize
        self.alreadyLoaded = False

    @define_scope#(initializer=tf.contrib.slim.xavier_initializer())
    def y_last(self):
        """
        Gets to logits by creating the computational graph to that point.
        Example: 

        Attributes:

        Args:
            
        Returns:
            
        TODO:
            1) 
        """
#         print(wtfStr)
        n_classes = int(self.yPlaceHolder.shape[1])
        x = self.xPlaceHolder
        # U = tf.get_variable(name = 'U', 
        #                     shape = (self.hyperParamDict['n_hidden'], n_classes), 
        #                 initializer = tf.contrib.layers.xavier_initializer())
        # bias = tf.get_variable(name = 'bias', shape = [n_classes], 
        #                        initializer = tf.constant_initializer(0))
        # sizeList = [self.hyperParamDict['n_hidden'], n_classes]
#         pretrainedEmbeddings = tf.Variable(self.embeddings)
        wordEmbeddings = tf.nn.embedding_lookup(params = self.pretrainedEmbeddings, ids = x)
        if self.chatty:
            print('shape of embeddings')
            print(wordEmbeddings.get_shape())
        #     print('U shape')
        #     print(U.get_shape())
        #     print('bias shape')
        #     print(bias.get_shape())
        y_last = LSTM(wordEmbeddings, self.sizeList, self.trueWordIdxs, self.outputKeepProb, self.inputKeepProb, 
            n_hidden = self.hyperParamDict['n_hidden'], num_layers = self.hyperParamDict['numLayers'],
            batch_size = self.hyperParamDict['batchSize'], max_length = self.maxLength, chatty = self.chatty)# TODO is y_last the correct thing to return?
        if self.chatty:
            print(y_last.get_shape())
        # print('como estas bitches')
        return(y_last)

    @define_scope
    def optimize(self):
        """
        Optimizer which uses adam optimization and gradient clipping
        Example: 

        Attributes:

        Args:
            

        Returns:
            
        TODO:
            1) 
        """
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
        """
        Loss function which uses cross entropy and sigmoid
        Example: 

        Attributes:

        Args:
            

        Returns:
            
        TODO:
            1) 
        """
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
        Saves the model weights. Not sure yet how to reload the model
        as a whole, but it's possible to load the weights and use them
        for prediction. Saves all the placeholders needed to run the model,
        weights and the trained embeddings.
        Example: 

        Attributes:

        Args:
            savePath (str): Where to save the model.
            session (tf session obj): The function used to save needs a 
                session....

        Returns:
            None
            
        TODO:
            1) 
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
        
        


def reloadModel(session, saverMetaPath, saverCheckPointPath):
    """
    Loades the model weights saved by he model.save() function.
    Example: 

    Attributes:

    Args:
        session (tf session obj): The function used to save needs a 
            session...
        saverMetaPath (str): Path to the .meta file
        saverCheckPointPath (str): Path to the folder which contains
            the checkpoint data.

    Returns:
        weightDict (dict): Just a dict that contains the weights, trained
            embeddings and the input placeholders
    TODO:
        1) 
    """
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