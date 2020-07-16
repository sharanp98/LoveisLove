
from model import SentimentModel
from transformers import AutoTokenizer,AutoModel
import pandas as pd
import csv
import torch

s = [ "Negative","Random","Positive"]


bert_model = AutoModel.from_pretrained('google/bert_uncased_L-4_H-256_A-4')
bert_tokenizer = AutoTokenizer.from_pretrained('google/bert_uncased_L-4_H-256_A-4')
model = SentimentModel(bert_model)
model.freeze_weights()
model.lstm.load_state_dict(torch.load('rnn.pth'))
model.layers.load_state_dict(torch.load('checkpoint.pth'))
model.eval()

f = open('result.csv','w',newline='') 
out_file = csv.writer(f)

data = pd.read_csv('results_ocr.csv')


out_file.writerow(['Filename','Category'])


for idx,row in data.iterrows():
    if type(row['Text']) == float:
        out_file.writerow([row['Filename'],"random"])
    else:
        scores = model([row['Text']],bert_tokenizer)
        sentiment = torch.argmax(torch.exp(scores),dim=1)
        out_file.writerow([row['Filename'],s[sentiment]])

f.close()


outputfile = pd.read_csv('result.csv')


submission = outputfile.sort_values(by='Filename')
submission.to_csv('submission.csv',index=False)





#print(data.head())


#text = [ "happy" ,
#        "of course gay men dress well. They didnt spend all that time in the closet doing nothing",
#      "Be calm because it is okay to be gay " , "it is very happy to know you are sad", "i am going to be sad",
#      "married with pride", "100 years of atlantic stories","we will not be erased","few say being LGBT is negative",
#      "Extremism in the defense of liberty is no vice. And moderation in the pursuit of justice is no virtue",
#      "All young people, regardless of sexual orientation or identity, deserve a safe and supportive environment in which to achieve their full potential.",
#      "Labels are for filing. Labels are for clothing. Labels are not for people.",
#      ]

#for t in text:
#    out=model([t],bert_tokenizer)
#    print(out)




