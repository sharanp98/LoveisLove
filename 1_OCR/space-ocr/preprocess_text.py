import pandas as pd

df = pd.read_csv('results.csv')
ls = []
for content in df['Text']:
    # print(type(content), content)
    ls.append(str(content).replace("\r\n", "")
              .encode("ascii", errors="ignore").decode())
df['Text'] = ls
df.to_csv('results_ascii.csv')
print(df)
