#!usr/bin/python3

# split.py
# Version 1.0.0
# 4-20-22

# Written By: Novia Wu

''' This is the script used to split data into classified subdirs
	as well as make a train and val portion of the training data. '''

import os
import shutil
import splitfolders			#type: ignore

# make new directory 'images'
new_path = r'images_classified' 
if not os.path.exists(new_path):
    os.makedirs(new_path)

# make 10 folders in 'images', each folder holds only pngs of the class denoted by folder name
curr_path = os.getcwd()
os.chdir(curr_path+"/"+new_path)
for i in range(1,11):
    # makeadir() evaluates your condition
    if True:
        path = '{}'.format(i)
        if not os.path.exists(path):
            os.mkdir(path)

# go through 'imgs' and move each png to its perspective sub-directory in 'images'            
os.chdir(curr_path+"/imgs")
path = os.getcwd()
for root, dirs, files in os.walk(path):
    for name in files:
        if not name[0] == '.':
            identifier = name[3]
            destination = curr_path+"/"+new_path+"/"
            dir = '10' if (identifier=='0') else identifier
            shutil.move(path+"/"+name, destination+dir+"/"+name)

# go back to root data folder and split all data into train(0.8) and valuation(0.2)  
os.chdir(curr_path)
splitfolders.ratio('images_classified', output="images_classified_split", seed=1337, ratio=(.8, 0.2)) 