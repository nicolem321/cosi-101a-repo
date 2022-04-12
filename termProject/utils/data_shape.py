#!usr/bin/python3

# data_shape.py
# Version 1.0.0
# 4/12/22

# Written By: Mason Ware

import pandas as pd

#TODO
#actually load images according to labels

class DataFrame():
    ''' A class to read txt data into a pandas dataframe and split it by class. '''
    def __init__(self, file_path: str='') -> None:
        self.in_file_path=file_path
        self.data_frame:"pd.DataFrame"
        self.class_series:"pd.Series"
    
    @classmethod
    def load_data(cls, file_path:str) -> "DataFrame":
        return cls(file_path)
    
    def get_df(self, csv_write:str='') -> "pd.DataFrame":
        ''' method to get a dataframe from given txt file. '''
        if csv_write:
            try:
                self.data_frame.to_csv(csv_write, index = None)
            except Exception as e:
                print(e)
        self.data_frame=pd.read_csv(self.in_file_path, delimiter = '\t')
        return(self.data_frame)
    
    def get_cls(self) -> "pd.Series":
        ''' method to get a series of each class from the dataframe. '''
        self.class_series=self.data_frame.groupby('class')['image_name'].apply(list)
        return(self.class_series)
    

if __name__ == '__main__':
    df = DataFrame()    # would normally be an import statment: `from utils.data_shape import DataFrame as df`
    data = df.load_data(file_path='termProject/data/train/labels.txt').get_df()
    print(data)
    