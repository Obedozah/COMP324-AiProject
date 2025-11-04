import arff
import pandas as pd

with open('datasets/KDDTrain+_20Percent.arff', 'r') as f:
    arff_file = arff.load(f)

df = pd.DataFrame(arff_file['data'], columns=[attr[0] for attr in arff_file['attributes']])

print(df.head())

print(df.tail())