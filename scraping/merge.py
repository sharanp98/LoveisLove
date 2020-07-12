# merge two dataframes
import pandas as pd
''' Do not uncomment these lines '''
# df1 = pd.read_csv('train.csv', index_col=0)
# df2 = pd.read_csv('train_2.csv', index_col=0)
df = pd.concat([df1, df2], axis=0)
df.reset_index(drop=True, inplace=True)

# df.to_csv('train.csv')
