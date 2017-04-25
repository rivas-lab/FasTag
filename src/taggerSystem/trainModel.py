import sys
import os
sys.path.append('src/taggerSystem/')
from simpleLSTMWithNNetModel import Model
import tensorflow as tf
import numpy as np
import time


def getBatch(x, y, trueWordIdxs, batch_size, batchNum):
    """
    Gets the next iteration in the batch for x and y and the indices of the true word
    Example: 

    Attributes:

    Args:
        x (tf placeholder): shape = (nObs, maxWordLength) holds all training data
        y (tf placeholder): shape = (nObs, nClasses) holds the response variable
        truWordIdxs (tf placeholder): shape = (nObs ,1) Holds index of last true word
            for a given note.
        batch_size (int): uhm... batch size.... yeah....
        batchNum (int): Which batch are we on. This with batch_size lets us calc
            the offset to be used for indexing.
        
    Returns:
        batch_x (tf placeholder): shape = (nObs, maxWordLength) holds all training data
        batch_y (tf placeholder): shape = (nObs, nClasses) holds the response variable
        batchTrueWordIdxs (tf placeholder): shape = (nObs ,1) Holds index of last true word
            for a given note.
        offset (int): index where the beginning of the batch was taken from
        
    TODO:
        1) Might wanna check that all obs are used only once.
    """
    offset = min((batchNum * batch_size), y.shape[0])
    batch_x = x[offset:(offset + batch_size), :]
    batch_y = y[offset:(offset + batch_size), :]
    batchTrueWordIdxs = trueWordIdxs[offset:(offset + batch_size)]
    # print('getting Batch')
    # print('total size {}. Last index of this batch {}'.format(x.shape[0], offset + batch_size))

    return(batch_x, batch_y, batchTrueWordIdxs, offset)

def trainModel(helperObj, embeddings, hyperParamDict, xDev, xTrain, yDev, yTrain, 
                lastTrueWordIdx_dev, lastTrueWordIdx_train, training_epochs,
               output_path, batchSizeTrain, sizeList, maxIncreasingLossCount = 3, batchSizeDev = 3295, chatty = False):
    """
    This function creates and runs the model. Everything needed for creating it, weights, hyper parameters, etc
    are passed in and the model is created accordingly. Then it is trained, and validated, and finally saved if a
    well performing model is found. it will stop running after the validation loss increases maxIncreasingLossCount
    times,  but this counter is reset after a new best model is found. Note that if the batch size for the training
    or testing is too large, or the max note length is too large this can hog memory and crash
    Example:
        helperObj (ModelHelper): Helper object which holds various information for the model. Check out
            src/taggerSystem/data_util.py for more information on this class
        embeddings (tf variable): Word embeddings used for this model.
        hyperParamDict (dict):  Dictionary which hods various tunable hyper parameters
        xDev (tf placeholder): placeholder for the validation set data. shape = (nObs_dev, maxNoteLength)
        xTrain (tf placeholder): placeholder for the training set data. shape = (nObs_train, maxNoteLength
        yTrain (tf placeholder): placeholder for the taining set ground truth. shape = (nObs_train, nClasses)
        yDev (tf placeholder): placeholder for the validation set ground truth. shape = (nObs_dev, nClasses
        lastTrueWordIdx_dev (tf placeholder): placeholder which holds indices for the last nonPadded word
            in the validation set
        lastTrueWordIdx_train (tf placeholder):placeholder which holds indices for the last nonPadded word
            in the trainig set
        training_epochs (int): Max number of epochs. Most often does not hit this number though
        output_path (str): Where to save literally everything. Model weights, predictions, data, etc
        batchSizeTrain (int): Size to use for stochastic gradient descent
        sizeList (list): This is important because the final layers weight sizes are stored here
        maxIncreasingLossCount (int): How soon after continuous increases in loss should the model stop.
            validation set loss is used for this.
        batchSizeDev (int): similar to batchSizeTrain except for the validation set. Important if you
            don't have a lot of memory and can't run it all at once.
        chatty (bool): How much printing should be done. right now it's only binary


    Attributes:

    Args:
        
    Returns:
        None: But it does save the model, it's predictions, trainig/dev loss, y data and true word indices.
        
    TODO:
        1) add example of hyperparameter dict.
    """
    totalBatchesDev = (xDev.shape[0]//batchSizeDev)
    total_batches = (xTrain.shape[0]//batchSizeTrain)
    epochAvgLoss = np.zeros(training_epochs)
    epochAvgLossValid = np.zeros(training_epochs)
    epochPredictions = np.zeros(shape = [training_epochs, yDev.shape[0], yDev.shape[1]])
    validLossIncreasingCount = 0
    # maxIncreasingLossCount = 3
    prevValidLoss = np.inf
    minValidLoss = np.inf
    tf.reset_default_graph()
    model = Model(nColsInput = helperObj.max_length, nLabels = helperObj.n_labels,
             embeddings = embeddings, hyperParamDict = hyperParamDict, sizeList = sizeList, chatty = chatty)
    # print('Here is xTrain')
    # print(xTrain)
    lastTrainObsIdx = xTrain.shape[0]
    lastTrainObsUsed = False
    with open(os.path.join(output_path, 'runOutput.txt'), 'w') as f:
        with tf.Session() as session:
            tf.global_variables_initializer().run()
            for epoch in range(training_epochs):
                print('***************************')
                print('***************************')
                print('Running on epoch %d'% (epoch))
                print('***************************')
                print('***************************')
                f.write("""***************************
                ***************************
                Running on epoch %d
                ***************************
                ***************************
                ***************************\n""" %(epoch))
                start = time.time()
                totalBatchError = 0.0
                avgBatchError = 0.0
                for batchNum in range(total_batches + 1):
                    # print("""
                    # Before doing any actual training be sure that you're using all training 
                    # data.
                    # """)
                    batch_x, batch_y, batchTrueWordIdxs, offset = getBatch(x = xTrain,
                                                                   y = yTrain,
                                                                   trueWordIdxs = lastTrueWordIdx_train,
                                                                  batch_size = batchSizeTrain,
                                                                  batchNum = batchNum)
                    if (offset + batchSizeTrain) >= lastTrainObsIdx:
                        lastTrainObsUsed = True
                    # print('Here is xBatch')
                    # print(batch_x)
                    _, batchError = session.run([model.optimize, model.loss_function], 
                                        feed_dict={model.xPlaceHolder: batch_x, 
                                        model.yPlaceHolder: batch_y, 
                                        model.trueWordIdxs: batchTrueWordIdxs,
                                        model.outputKeepProb: hyperParamDict['outputKeepProb'], 
                                        model.inputKeepProb: hyperParamDict['inputKeepProb']})
                    totalBatchError += batchError
                    if(batchNum%25 == 0):
                        # f.write('running iteration %d with loss %3f \n'% (b, c))
                        # print('Total run time was %3f'% (time.time() - start))
                        print('running iteration %d with loss %3f at time %3f'% (batchNum, batchError, (time.time() - start)))
                avgBatchError = totalBatchError/total_batches
    #             batchSizeDev = 3295
    #             totalBatchesDev = (xDev.shape[0]//batchSizeDev)
                pred_y = np.full(shape = yDev.shape, fill_value = -1.0, dtype = np.float32)
                # print(' here is xDev')
                # print(xDev)
                if not lastTrainObsUsed:
                    print('Never used the last training observation.')
                    print('exiting now')
                    1/0
                for batchNum in range(totalBatchesDev + 1):
    #                 print(b)
                    devBatch_x, _, devBatchTrueWordIdxs, offset = getBatch(x = xDev,
                                                                   y = yDev,
                                                                   trueWordIdxs = lastTrueWordIdx_dev,
                                                                  batch_size = batchSizeDev,
                                                                  batchNum = batchNum)
                    # print('here is xDev batch')
                    # print(devBatch_x)
                    pred_yBatch = session.run(model.y_last,feed_dict={model.xPlaceHolder: devBatch_x,
                                          model.trueWordIdxs:devBatchTrueWordIdxs,
                                          model.outputKeepProb: 1,
                                          model.inputKeepProb: 1}, )# must be set to one for predictions.
                    pred_y[offset:(offset + batchSizeDev), :] = pred_yBatch
                if (pred_y == -1).all():
                    print('negative values exist. This means indexing is off in pred_yBatch')
                    1/0
                validLoss = tf.nn.sigmoid_cross_entropy_with_logits(logits = pred_y, 
                                                     labels = tf.cast(yDev, tf.float32))
    #             print('validLoss')
                validLoss = tf.reduce_mean(validLoss).eval()
    #             validLoss = validLoss.eval()
                epochPredictions[epoch,:,:] = pred_y
                epochAvgLossValid[epoch] = validLoss
                epochAvgLoss[epoch] = totalBatchError/total_batches
                f.write('average training loss %f \n'% (epochAvgLoss[epoch]))
                f.write('test loss %f \n'%(validLoss))
                f.write('Total run time was %3f \n'% (time.time() - start))
                print('average training loss %f'% (epochAvgLoss[epoch]))
                print('test loss %f'%(validLoss))
    #             print('previous valid loss %f'%(prevValidLoss))
                print('Total run time was %3f'% (time.time() - start))
    #             f.write('average loss %f \n'%(totalError/total_batches)) 
    #             f.write('test loss %f \n'%(validLoss))
    #             f.write('Total run time was %3f \n'% (time.time() - start))
                if validLoss <= minValidLoss:
                    validLossIncreasingCount = 0
                    minValidLoss = validLoss
                    f.write('New best model found. Saving\n')
                    print('New best model found. Saving')
                    print(output_path)
                    model.save(session = session, savePath = os.path.join(output_path, 'bestModel'))
    #                 all_saver.save(session, os.path.join(modelRunOutputPath, 'bestModel'))
                    # save model
                if validLoss > prevValidLoss:
                    print('validation Loss Increase')
                    validLossIncreasingCount += 1
    #             print('increasing loss count %d'%(validLossIncreasingCount))
                prevValidLoss = validLoss
                if validLossIncreasingCount == maxIncreasingLossCount:
                    print('Stopping early because of increasing validation loss')
                    f.write('Stopping early because of increasing validation loss\n')
                    # model.save(session = session, savePath = output_path)
    #                 f.write('Stopping early because of increasing validation loss')
                    break
                # 1/0
            lastEpoch = epoch
    epochPredictions = epochPredictions[0:(lastEpoch + 1), :, :]
    # print(epochPredictions)
    # print(epochPredictions.shape)
    epochAvgLoss = epochAvgLoss[0:(lastEpoch + 1)]
    epochAvgLossValid = epochAvgLossValid[0:(lastEpoch + 1)]
    np.savetxt(os.path.join(output_path, 'epochPreds.gz'), epochPredictions.flatten())
    np.savetxt(os.path.join(output_path, 'epochAvgLoss.gz'), epochAvgLoss)
    np.savetxt(os.path.join(output_path, 'epochAvgLossValid.gz'), epochAvgLossValid)
    np.savetxt(os.path.join(output_path, 'epochPredsShape.gz'), 
                        np.array(epochPredictions.shape), fmt='%i')
    np.savetxt(os.path.join(output_path, 'yDev.gz'),yDev)
    #np.save(os.path.join(modelRunOutputPath, 'xDev.npy'), xDev)
    np.savetxt(os.path.join(output_path, 'devTrueIdxs.gz'), lastTrueWordIdx_dev)
                