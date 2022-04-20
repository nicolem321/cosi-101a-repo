#!usr/bin/python3

# data_shape.py
# Version 1.0.0
# 4/12/22

# Written By: Mason Ware

''' This is a python module to read a txt file of the following line-by-line format: 'img.png    class#' and return dataframes
    for different dissections of the data. The following is a generic use case of the module in code:

    from utils.data import DataPy as dpy

    dpy.read_data(dpy, 'path_name.txt')

    data_frame = dpy.get_df()        # this gets a generic two dimensional dataframe of each image and its class
    class_frame = dpy.get_class()    # this gets a two dimensional dataframe of each class and its set of images
    # more coming potentially ... 

    dpy.to_csv('file_path')          # this will write the last dataframe you created with the data to csv format

    To note: this module is local-system specific, meaning that it will read the local system path of your machine and redirect
    to the image folder to replace a name such as '0001.png' with '<local_path_name>data/train/imgs/0001.png' so that when iterated 
    over, it can be plugged directly into a call. That means that a uniform file hierarchy for this script and the data folder is 
    necessary for this to work. If you have pulled straight from the github, then make sure that you have a data folder with a train
    subdir that contains an imgs subdir with all the .pngs in it as well as a file named labels.txt. Lastly, make sure this script
    is in the utils folder and is named data.py. '''

import os

import pandas as pd
import colorama                         # type: ignore
from colorama import Fore               # type: ignore


class DataPy():
    ''' A class to read txt data into a pandas dataframe. '''
    def __init__(self) -> None:
        self.in_file_path:str
        self.curr_frame:"pd.DataFrame"
    
    def read_data(self, in_file_path) -> None:
        self.in_file_path=in_file_path
        
    def get_df(self) -> "pd.DataFrame":
        ''' method to generate a pandas dataframe from given txt file. '''
        try:
            self.curr_frame = pd.read_csv(self.in_file_path, delimiter = '\t')     # txt tab spaced doc expected
            old_path = os.getcwd()
            path_lst = old_path.split('/')
            if path_lst[len(path_lst)-1] == 'termProject':
                os.chdir('data/train/imgs')
            elif path_lst[len(path_lst)-1] == 'utils':
                os.chdir('../data/train/imgs')
            for index, row in self.curr_frame.iterrows():
                # path_name = os.getcwd() + '/data/train/imgs/' + row['image_name']       # mac
                path_name = os.getcwd() + '\\data\\train\\imgs\\' + row['image_name']       # windows
                self.curr_frame.loc[index, 'image_name'] = path_name
            os.chdir(old_path)
            print(Fore.RED + f'\n=====>DATA:\n')
            print(Fore.RESET + f'{self.curr_frame}')
        except Exception as e:
            print(f'\n\n{e}\n\n')
        return(self.curr_frame)
    
    def get_cls(self) -> "pd.DataFrame":
        ''' method to get a dataframe organized by class. '''
        try:
            self.curr_frame = pd.read_csv(self.in_file_path, delimiter = '\t').groupby('class')['image_name'].apply(list).to_frame()     # txt tab spaced doc expected
            old_path = os.getcwd()
            path_lst = old_path.split('/')
            if path_lst[len(path_lst)-1] == 'termProject':
                os.chdir('data/train/imgs')
            elif path_lst[len(path_lst)-1] == 'utils':
                os.chdir('../data/train/imgs')
            for index, row in self.curr_frame.iterrows():
                for i in range(len(row['image_name'])):
                    # path_name = os.getcwd() + '/data/train/imgs/' + row['image_name']       # mac
                    path_name = os.getcwd() + '\\data\\train\\imgs\\' + row['image_name']       # windows
                    row['image_name'][i] = path_name
            os.chdir(old_path)
            print(Fore.RED + f'\n\n=====>CLASSES:')
            print(Fore.RESET + f'{self.curr_frame}\n')
        except Exception as e:
            print(f'\n\n{e}\n\n')
        return(self.curr_frame)
    
    def to_csv(self, out_file_path) -> None:
        ''' method to write to a csv file. '''
        try:
            self.curr_frame.to_csv(out_file_path, index = None)
        except Exception as e:
            print(f'\n\n{e}\n\n')
            


if __name__ == '__main__':
    dpy = DataPy()    # would normally be an import statment: `from utils.data import DataPy as dtdf
    dir_str = os.path.normpath(os.getcwd() + os.sep + os.pardir) + '\data\\train\labels.txt'
    dpy.read_data(dir_str)
    data_df = dpy.get_df()
    class_df = dpy.get_cls()
    
    # dpy.to_csv('/Users/masonware/Desktop/COSI_101A/termProject/data/train/test.csv')
    
    