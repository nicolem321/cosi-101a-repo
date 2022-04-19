#!usr/bin/python3

# main.py
# Version 1.0.0
# 4-19-22

# Written By: Mason Ware 


''' This file is a single representation of part of the code that appears in the 
    file ./hw6.ipynb '''


import os
from typing import Tuple

from utils import timer

import idx2numpy                                            # type: ignore
import numpy as np                                          # type: ignore
from mlxtend.data import loadlocal_mnist                    # type: ignore
from numpy import mean                                      # type: ignore
from numpy import std                                       # type: ignore
from matplotlib import pyplot as plt                        # type: ignore
from sklearn.model_selection import KFold                   # type: ignore
from tensorflow.keras.utils import to_categorical           # type: ignore
from tensorflow.keras.models import Sequential              # type: ignore
from tensorflow.keras.layers import Conv2D                  # type: ignore
from tensorflow.keras.layers import MaxPooling2D            # type: ignore
from tensorflow.keras.layers import Dense                   # type: ignore
from tensorflow.keras.layers import Flatten                 # type: ignore
from tensorflow.keras.optimizers import SGD                 # type: ignore


train_img_file = f'{os.getcwd()}/data/emnist-digits-train-images-idx3-ubyte'
test_img_file = f'{os.getcwd()}/data/emnist-digits-test-images-idx3-ubyte'
train_label_file = f'{os.getcwd()}/data/emnist-digits-train-labels-idx1-ubyte'
test_label_file = f'{os.getcwd()}/data/emnist-digits-test-labels-idx1-ubyte'

X_train = idx2numpy.convert_from_file(train_img_file)
y_train = idx2numpy.convert_from_file(train_label_file)
X_test = idx2numpy.convert_from_file(test_img_file)
y_test = idx2numpy.convert_from_file(test_label_file)

# # visualize data
# for i in range(9): 
#     plt.subplot(330 + 1 + i) 
#     plt.imshow(X_train[i], cmap=plt.get_cmap('gray'))
# plt.show()

@timer
# load train and test dataset
def load_dataset() -> Tuple:
    # reshape dataset to have a single channel
    trainX = X_train.reshape((X_train.shape[0], 28, 28, 1))
    testX = X_train.reshape((X_train.shape[0], 28, 28, 1))
    
    # one hot encode target values
    trainY = to_categorical(y_train)
    testY = to_categorical(y_test)
    return (trainX, trainY, testX, testY)

@timer
# scale pixels
def prep_pixels(train, test):
	# convert from integers to floats
    train_norm = train.astype('float32')
    test_norm = test.astype('float32')
    # normalize to range 0-1
    train_norm = train_norm / 255.0
    test_norm = test_norm / 255.0
    # return normalized images
    return train_norm, test_norm

@timer
# define cnn model
def define_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(10, activation='softmax'))
    # compile model
    opt = SGD(learning_rate=0.01, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

@timer
# evaluate a model using k-fold cross-validation
def evaluate_model(dataX, dataY, n_folds=5):
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
# plot diagnostic learning curves
def summarize_diagnostics(histories):
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
# summarize model performance
def summarize_performance(scores):
	# print summary
    print('Accuracy: mean=%.3f std=%.3f, n=%d' % (mean(scores)*100, std(scores)*100, len(scores)))
    # box and whisker plots of results
    plt.boxplot(scores)
    plt.show()

@timer
# run the test harness for evaluating a model
def run_test_harness():
    #load data set
    trainX, trainY, testX, testY = load_dataset()
    # prepare pixel data
    trainX, testX = prep_pixels(trainX, testX)
    print('pixels prepped!!')
    # evaluate model
    scores, histories = evaluate_model(trainX, trainY)
    # learning curves
    summarize_diagnostics(histories)
	# summarize estimated performance
    summarize_performance(scores)
 

run_test_harness()