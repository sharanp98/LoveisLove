# merge three dataframes
import pandas as pd

df1 = pd.read_csv('train_negative.csv', index_col=0)
df2 = pd.read_csv('train_positive.csv', index_col=0)
df3 = pd.read_csv('train_neutral.csv', index_col=0)

df = pd.concat([df1, df2, df3], axis=0)
df.reset_index(drop=True, inplace=True)

df.to_csv('train.csv')
