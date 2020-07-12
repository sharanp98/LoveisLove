import pandas as pd
df = pd.read_csv("scraping/train.txt", delimiter="\n",
                 names=['Quote'])
df['Sentiment'] = [0]*len(df)

df.to_csv('train_2.csv')
