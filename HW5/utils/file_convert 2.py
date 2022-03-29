#! usr/bin/python3
import csv

class FileConverter:
    '''Class to convert a txt file to a csv file.'''
    def __init__(self, filename: str) -> None:
        self.file_name = filename
        self.rows = []

    def insert_commas(self) -> None:
        '''method to insert commas into a txt file with no commas.'''
        with open(self.file_name, encoding='UTF-8') as f:
            for line in f:
                self.rows.append(line.split('\t'))

    def clean_matrix(self) -> None:
        '''method to clean the matrix of any newlines or unwanted 
        characters before writing to csv'''
        for row in self.rows:
            for item in row:
                if item == '\n':
                    row.remove(item)
                
    def load_csv(self, output_file_name: str) -> None:
        '''method to use a 2d matrix of rows of data to build a csv file.'''
        # output_file_name = 'hw5_data/training.csv'
        with open(output_file_name, 'w') as out:
            write = csv.writer(out)
            write.writerows(self.rows)
        print('\n\n\n', f'='*50, f'\n\nCSV CREATED AT: \n( {output_file_name} )\n\njob finished successfully :)\n', f'='*50, '\n\n\n')
            
            
if __name__ == '__main__':
    # input_file_name = input('Enter .txt File Path > ')
    input_file = input('Input File Path > ')
    fc = FileConverter(filename=input_file)
    fc.insert_commas()
    fc.clean_matrix()
    destination = input('Output File Path: > ')
    fc.load_csv(output_file_name=destination)