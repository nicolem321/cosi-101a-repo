import os
import platform

import numpy as np
from mlxtend.data import loadlocal_mnist        # type: ignore


X, y = loadlocal_mnist(
            images_path=f'{os.getcwd()}/data/emnist-digits-train-images-idx3-ubyte', 
            labels_path=f'{os.getcwd()}/data/emnist-digits-train-labels-idx1-ubyte')


import numpy as np
import idx2numpy                                # type: ignore
import matplotlib.pyplot as plt

train_img_file = f'{os.getcwd()}/data/emnist-digits-train-images-idx3-ubyte'
test_img_file = f'{os.getcwd()}/data/emnist-digits-test-images-idx3-ubyte'
train_label_file = f'{os.getcwd()}/data/emnist-digits-train-labels-idx1-ubyte'
test_label_file = f'{os.getcwd()}/data/emnist-digits-test-labels-idx1-ubyte'

X_train = idx2numpy.convert_from_file(train_img_file)
y_train = idx2numpy.convert_from_file(train_label_file)

X_test = idx2numpy.convert_from_file(test_img_file)
y_test = idx2numpy.convert_from_file(test_label_file)

print(y_train[0:9])
for i in range(9): 
    plt.subplot(330 + 1 + i) 
    # plt.legend(y_train)
    plt.imshow(X_train[i], cmap=plt.get_cmap('gray'))
plt.show()
