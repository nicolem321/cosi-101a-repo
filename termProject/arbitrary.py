import os
train_1_dir = os.path.join('data/train/imgs_classified_split/train/1')
train_2_dir = os.path.join('data/train/imgs_classified_split/train/2')
train_3_dir = os.path.join('data/train/imgs_classified_split/train/3')
train_4_dir = os.path.join('data/train/imgs_classified_split/train/4')
train_5_dir = os.path.join('data/train/imgs_classified_split/train/5')
train_6_dir = os.path.join('data/train/imgs_classified_split/train/6')
train_7_dir = os.path.join('data/train/imgs_classified_split/train/7')
train_8_dir = os.path.join('data/train/imgs_classified_split/train/8')
train_9_dir = os.path.join('data/train/imgs_classified_split/train/9')
train_10_dir = os.path.join('data/train/imgs_classified_split/train/10')

valid_1_dir = os.path.join('data/train/imgs_classified_split/test/1')
valid_2_dir = os.path.join('data/train/imgs_classified_split/test/2')
valid_3_dir = os.path.join('data/train/imgs_classified_split/test/3')
valid_4_dir = os.path.join('data/train/imgs_classified_split/test/4')
valid_5_dir = os.path.join('data/train/imgs_classified_split/test/5')
valid_6_dir = os.path.join('data/train/imgs_classified_split/test/6')
valid_7_dir = os.path.join('data/train/imgs_classified_split/test/7')
valid_8_dir = os.path.join('data/train/imgs_classified_split/test/8')
valid_9_dir = os.path.join('data/train/imgs_classified_split/test/9')
valid_10_dir = os.path.join('data/train/imgs_classified_split/test/10')

train_1_names = os.listdir(train_1_dir)
train_2_names = os.listdir(train_2_dir)
train_3_names = os.listdir(train_3_dir)
train_4_names = os.listdir(train_4_dir)
train_5_names = os.listdir(train_5_dir)
train_6_names = os.listdir(train_6_dir)
train_7_names = os.listdir(train_7_dir)
train_8_names = os.listdir(train_8_dir)
train_9_names = os.listdir(train_9_dir)
train_10_names = os.listdir(train_10_dir)



# print(train_1_names[:10])
# print('total training 1 images:', len(os.listdir(train_1_dir)))


# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg

# # Parameters for our graph; we'll output images in a 4x4 configuration
# nrows = 10
# ncols = 10

# # Index for iterating over images
# pic_index = 0
# # Set up matplotlib fig, and size it to fit 4x4 pics
# fig = plt.gcf()
# fig.set_size_inches(ncols * 4, nrows * 4)

# pic_index += 8
# next_1_pic = [os.path.join(train_1_dir, fname) 
#                 for fname in train_1_names[pic_index-8:pic_index]]
# next_2_pic = [os.path.join(train_2_dir, fname) 
#                 for fname in train_2_names[pic_index-8:pic_index]]
# next_3_pic = [os.path.join(train_3_dir, fname) 
#                 for fname in train_3_names[pic_index-8:pic_index]]
# next_4_pic = [os.path.join(train_4_dir, fname) 
#                 for fname in train_4_names[pic_index-8:pic_index]]
# next_5_pic = [os.path.join(train_5_dir, fname) 
#                 for fname in train_5_names[pic_index-8:pic_index]]
# next_6_pic = [os.path.join(train_6_dir, fname) 
#                 for fname in train_6_names[pic_index-8:pic_index]]
# next_7_pic = [os.path.join(train_7_dir, fname) 
#                 for fname in train_7_names[pic_index-8:pic_index]]
# next_8_pic = [os.path.join(train_8_dir, fname) 
#                 for fname in train_8_names[pic_index-8:pic_index]]
# next_9_pic = [os.path.join(train_9_dir, fname) 
#                 for fname in train_9_names[pic_index-8:pic_index]]
# next_10_pic = [os.path.join(train_10_dir, fname) 
#                 for fname in train_10_names[pic_index-8:pic_index]]

# for i, img_path in enumerate(next_1_pic+next_2_pic+next_3_pic+next_4_pic+next_5_pic+next_6_pic+next_7_pic+next_8_pic+next_9_pic+next_10_pic):
#   # Set up subplot; subplot indices start at 1
#   sp = plt.subplot(nrows, ncols, i + 1)
#   sp.axis('Off') # Don't show axes (or gridlines)

#   img = mpimg.imread(img_path)
#   plt.imshow(img)

# plt.show()

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# All images will be rescaled by 1./255
train_datagen = ImageDataGenerator(rescale=1/255)
validation_datagen = ImageDataGenerator(rescale=1/255)

# Flow training images in batches of 120 using train_datagen generator
train_generator = train_datagen.flow_from_directory(
        'data/train/imgs_classified_split/train/',  # This is the source directory for training images
        classes = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'],
        target_size=(200, 200),  # All images will be resized to 200x200
        batch_size=85,
        # Use binary labels
        class_mode='binary')

# Flow validation images in batches of 19 using valid_datagen generator
validation_generator = validation_datagen.flow_from_directory(
        'data/train/imgs_classified_split/val/',  # This is the source directory for training images
        classes = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'],
        target_size=(200, 200),  # All images will be resized to 200x200
        batch_size=20,
        # Use binary labels
        class_mode='binary',
        shuffle=False)


import tensorflow as tf
import numpy as np
from itertools import cycle

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from sklearn.metrics import roc_auc_score

model = tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape = (200,200,3)), 
                                    tf.keras.layers.Dense(128, activation=tf.nn.relu), 
                                    tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)])

model.summary()

model.compile(optimizer = tf.optimizers.Adam(),
              loss = 'binary_crossentropy',
              metrics=['accuracy'])

#steps to 9 with 
history = model.fit(train_generator,
      steps_per_epoch=8,  
      epochs=15,
      verbose=1,
      validation_data = validation_generator,
      validation_steps=8)