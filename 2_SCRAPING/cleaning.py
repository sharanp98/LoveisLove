import re

res = []

with open('train_negative.csv', 'r') as f:
    for line in f.readlines():
        x = re.sub('\W+', ' ', line[3:-3])
        res.append(line[:3]+x+line[-3:])

with open('train_negative_clean', 'w') as f:
    for item in res:
        f.write("%s\n" % item)
