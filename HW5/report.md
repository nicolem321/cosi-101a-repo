_Written By: Mason Ware_

HW5
===

COSI 101A -- Brandeis University
================================
Mason W.

##### Professor Pengyu Hong

###### Date: 03/26/22

* * *

### Description

The objective of this HW was to build a K-nearest-neighbors classifier using the training samples, and then to test it on the test samples.

The KNN model is built, fit, and run in `/hw5_knn.ipynb`. The strategy I took to execute this was to import scikit-learn's `KNeighborsClassifier` as my model as well as some of their other tools for both smoothing data and for evaluating the model's performance.


* * *

### The Jupyter Notebook

The entirety of the project can be found within the file `/hw5_knn.ipynb`. This file is a jupyter notebook separated into 6 main sections:

* Imports

This is the section where importing is done. All relevant data managment packages are imported as well as the scikit-learn packages discussed earlier.

* Datasets

This is the section in which the data sets are constructed. Mainly, we are given three pieces of the puzzle: a training dataset, a training set of labels, and a test set of unlabeled data. That being said, two of those pieces of data (the train data and train labels) are combined in one text file: `data_train.txt`. Within the datasets section, this training data set is split into X_train and y_train. One other important thing to note is that data is given to us in txt format, meaning any interaction has to be done on a solely grep-basis. Therefore, within the subdirectory `utils`, I wrote a python module named `file_convert.py` that accepts a txt file like the ones found at `data_train.txt` and `data_test.txt` and returns a csv file with the same contents. These files were converted to csv files using this method and saved at `data/testing.csv` and `data/training.csv`, respectively. Those are the files interacted with. 

I label the '3 pieces of the puzzle' as X_train, y_train and X_test, respectively. The foruth piece is y_test, or the testing labels. In other words, they are our results: the labels our model will generate based on the unlabeled testing data.

* * Scaling Data

This is a subsection to the datasets section. Within this section, the data is scaled so that it is more presentable and so that the each feautre contibutes approximately proportionately to the final distance between neighbors. The scaling is perfmored using scikit-learn's `StandardScaler`.

* Training the KNN Model

This section works to train the model and to see if it is capable of producing results that we are happy with. It instantiates a new instance of the model and fits it on the training data and training labels. Once this is done, the predicitons are recieved and printed.

* Model Evaluation

In this section, I pick which distance metric is best for our dataset by evaluating each run on the default settings save their distance metric. Each trial is evaluated using certain metrics including the accuracy score, the recall score, and the f1 score. Using scikitlearn's `confusion_matrix` and `classification_report` methods, I am able to generate a helpful and informative report of the models performance. The three metrics I tried were: `manhattan` (city block), `minkowski` (cosine), and `euclidean`. Based on the metrics produced, the city-block distance metric worked best, achieving a 94% across the board, on average 1-2% higher than the other metrics.

* Picking The Right K

Where in previous sections, the model was run with a standard k=5, this section serves to pick an ideal value of k for our data set based on the minimal amount of errors assosiated with each value. This section runs the data on 40 different potential k's and plots the results gainst the error values.

Using this graph, I pick the k value of 3 because, other than the obvious 1, it provides as the lowest error.

* Running KNN On The Test Set

This section serves to actually accomplish the goal of this assignment and obtain labels from our test data. Therefore, our model is created with a k of 3. It is fit to the training data and then operated on the testing data. The resulting predictions are written to a txt file and a csv file, both entitled `data/results.*`.


* * *

### Supplemental Code

There are other files and directories in this project, specifically, `utils/` and `testKNN.m`. Utils houses a file: `file_convert.py` which has already been discussed. The matlab file is irrelevant to the scope of this project and was sourced from a class lecture. 



_© 2022 MASON WARE_
