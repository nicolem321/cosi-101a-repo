#!usr/bin/python3

# main.py
# Version 2.0.0
# 4-19-22

# Written By: Mason Ware 


''' This file follows the structure of my main.py file in my HW6 submission, with
	additions. '''


import argparse
import os
import subprocess
from typing import Tuple, Iterable
import unittest

from utils.timer import timer

# import idx2numpy                                            # type: ignore
import numpy as np                                          # type: ignore
from numpy import mean                                      # type: ignore
from numpy import std                                       # type: ignore
from matplotlib import pyplot as plt                        # type: ignore
import matplotlib.image as mpimg                            # type: ignore
from sklearn.model_selection import KFold                   # type: ignore
from tensorflow.keras.utils import to_categorical           # type: ignore
from tensorflow.keras.models import Sequential              # type: ignore
from tensorflow.keras.layers import Conv2D                  # type: ignore
from tensorflow.keras.layers import MaxPooling2D            # type: ignore
from tensorflow.keras.layers import Dense                   # type: ignore
from tensorflow.keras.layers import Flatten                 # type: ignore
from tensorflow.keras.optimizers import SGD                 # type: ignore
# from utils.data import DataPy as dpy


class Evaluation:
    ''' This is a class to run a projected evaluation of a cnn with varying differences. '''
    def __init__(self, X_train: "np.array", y_train: "np.array", X_test: "np.array", y_test: "np.array", k: int) -> None:
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.k = k
        
    @timer
    def load_dataset(self):
        ''' load train and test dataset. '''
        # reshape dataset to have a single channel
        trainX = self.X_train.reshape((self.X_train.shape[0], 28, 28, 1))
        testX = self.X_test.reshape((self.X_test.shape[0], 28, 28, 1))
        # one hot encode target values
        trainY = to_categorical(self.y_train)
        testY = to_categorical(self.y_test)
        return (trainX, trainY, testX, testY)

    @timer
    def prep_pixels(self, train, test):
        ''' scale pixels. '''
        # convert from integers to floats
        train_norm = train.astype('float32')
        test_norm = test.astype('float32')
        # normalize to range 0-1
        train_norm = train_norm / 255.0
        test_norm = test_norm / 255.0
        # return normalized images
        return train_norm, test_norm

    @timer
    def define_model():
        ''' define cnn model. '''
        model = Sequential()
        model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
        model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Flatten())
        model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(10, activation='softmax'))
        # compile model
        opt = SGD(learning_rate=0.01, momentum=0.9)
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    @timer
    def evaluate_model(dataX, dataY, n_folds=5):                    #! change n-folds here
        ''' evaluate a model using k-fold cross-validation. '''
        scores, histories = list(), list()
        # prepare cross validation
        kfold = KFold(n_folds, shuffle=True, random_state=1)
        # enumerate splits
        for train_ix, test_ix in kfold.split(dataX):
            # define model
            model = define_model()
            # select rows for train and test
            trainX, trainY, testX, testY = dataX[train_ix], dataY[train_ix], dataX[test_ix], dataY[test_ix]
            # fit model
            history = model.fit(trainX, trainY, epochs=10, batch_size=32, validation_data=(testX, testY), verbose=0)
            # evaluate model
            _, acc = model.evaluate(testX, testY, verbose=0)
            print('> %.3f' % (acc * 100.0))
            # stores scores
            scores.append(acc)
            histories.append(history)

        return scores, histories

    @timer
    def summarize_diagnostics(histories):
        ''' plot diagnostic learning curves. '''
        for i in range(len(histories)):
            # plot loss
            plt.subplot(2, 1, 1)
            plt.title('Cross Entropy Loss')
            plt.plot(histories[i].history['loss'], color='blue', label='train')
            plt.plot(histories[i].history['val_loss'], color='orange', label='test')
            # plot accuracy
            plt.subplot(2, 1, 2)
            plt.title('Classification Accuracy')
            plt.plot(histories[i].history['accuracy'], color='blue', label='train')
            plt.plot(histories[i].history['val_accuracy'], color='orange', label='test')
        plt.show()
    
    @timer
    def summarize_performance(scores):
        ''' summarize model performance. '''
        # print summary
        print('Accuracy: mean=%.3f std=%.3f, n=%d' % (mean(scores)*100, std(scores)*100, len(scores)))
        # box and whisker plots of results
        plt.boxplot(scores)
        plt.show()
    
    
    
    

if __name__ == "__main__":
    # generate data sets
    lstx_train, lsty_train, lstx_test, lsty_test = list(), list(), list(), list()
    for root, dirs, files in os.walk("data/train/imgs_classified_split", topdown=False):
        # build train and val data sets
        for name in files:
            cur_path = (os.path.join(root, name)).split('/')[3:]
            if not cur_path[len(cur_path)-1][0] == '.':
                if cur_path[0]=='train':
                    lstx_train.append(mpimg.imread(os.getcwd() + '/' + os.path.join(root, name)))
                    lsty_train.append(cur_path[1])
                    # need to add the label as well
                elif cur_path[0]=='val':
                    lstx_test.append(mpimg.imread(os.getcwd() + '/' + os.path.join(root, name)))
                    lsty_test.append(cur_path[1])
                    # need to add the label as well
    X_train = np.array(lstx_train)
    y_train = np.array(lsty_train)
    X_test = np.array(lstx_test)
    y_test = np.array(lsty_test)
    
    
    parser = argparse.ArgumentParser(description="Handwriting Recognition CNN")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()

    if args.eval:
        eval = Evaluation(X_train=X_train, 
                          y_train=y_train, 
                          X_test=X_test, 
                          y_test=y_test, 
                          k=5)                                                          # The higher the k value, the longer the runtime
        #load data set
        trainX, trainY, testX, testY = eval.load_dataset()
        # prepare pixel data
        trainX, testX = eval.prep_pixels(trainX, testX)
        # evaluate model
        scores, histories = eval.evaluate_model(trainX, trainY)
        # learning curves
        eval.summarize_diagnostics(histories)
        # summarize estimated performance
        eval.summarize_performance(scores)
        
        # subprocess.call('./build.sh', shell=False)
    if args.test:
        # write test suite and switch below
        
        print(f'-'*60,f'\nRunning Test Suite...\n\n', f'-'*60)
        suite = unittest.TestLoader().loadTestsFromModule(test_hw5)
        unittest.TextTestRunner(verbosity=3).run(suite)
        
    if args.run:
        
        app.run(debug=True, port=5000)

####################################################################################################
####################################################################################################

# TODO
# final model to come...
# create a final model class with the code to come...