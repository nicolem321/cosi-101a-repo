import pandas as pd

in_file_path = 'termProject/data/train/labels.txt'
out_file_path = 'termProject/data/labels.csv'
df = pd.read_csv(in_file_path, delimiter = '\t')
df.to_csv(out_file_path, index = None)

