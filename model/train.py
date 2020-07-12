import torch
import torch.nn as nn
from torch.utils.data import DataLoader , Dataset,SubsetRandomSampler
import pandas as pd
from transformers import AutoTokenizer,AutoModel
import numpy as np
from model import SentimentModel

class SentimentDataSet(Dataset):

    
    def __init__(self,csv_file):

        self.sentiment_file = pd.read_csv(csv_file) 

    def __len__(self):

        return len(self.sentiment_file)

    def __getitem__(self,idx):

        sentiment_text = str(self.sentiment_file.iloc[idx,1])
        sentiment = int(self.sentiment_file.iloc[idx,0])

        ret_data = {'text': sentiment_text, 'sentiment':sentiment }

        return ret_data




def train_model(model,train,valid,optimizer,tokenizer):
    
    criterion = nn.CrossEntropyLoss()
    epochs = 20
    best_val_loss = 100
    for e in range(epochs):
        epoch_loss = 0
        for data in  train:
            model.zero_grad()
            optimizer.zero_grad()
            scores = model(data['text'],tokenizer)
            loss = criterion(scores,data['sentiment'])

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        model.eval()
        val_loss =0
        for data in valid:
            scores = model(data['text'],tokenizer)
            val_loss += criterion(scores,data['sentiment'])

        final_loss = val_loss.item() / len(valid)
        print("---------EPOCH "+str(e) + "----------\n")
        print("Training Loss=" + str(epoch_loss/len(train)) + "\n" )
        print("Validation Loss = " + str(final_loss)+"\n")

        if(final_loss < best_val_loss):
            print("Saving Model\n")
            torch.save(model.state_dict(),'checkpoint.pth')
            best_val_loss = final_loss




dataset = SentimentDataSet('formatted_data.csv')
dataset_size = len(dataset)
#print(dataset_size)
dataset_indices = list(range(dataset_size))
np.random.shuffle(dataset_indices)


val_index = int(np.floor(0.32*dataset_size))

train_idx , val_idx = dataset_indices[val_index:] , dataset_indices[:val_index]
train_sampler = SubsetRandomSampler(train_idx)
val_sampler = SubsetRandomSampler(val_idx)


train_dataloader = DataLoader(dataset,shuffle=False,batch_size=10,sampler=train_sampler)
val_loader = DataLoader(dataset,shuffle=False,batch_size=10,sampler=val_sampler)

#print(next(iter(dataloader)))

bert_model = AutoModel.from_pretrained('google/bert_uncased_L-4_H-256_A-4')
bert_tokenizer = AutoTokenizer.from_pretrained('google/bert_uncased_L-4_H-256_A-4')
model = SentimentModel(bert_model.embeddings.word_embeddings)
#model.freeze_weights()


optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
train_model(model,train_dataloader,val_loader,optimizer,bert_tokenizer)



