import torch
import torch.nn as nn
import torch.nn.functional as F

class SentimentModel(nn.Module):

    def __init__(self,bert):
        
        super().__init__()
        self.bert = bert
        #self.layers = nn.Sequential(
        #            nn.Linear(256,128),
        #            nn.ReLU(),
        #            nn.Linear(128,64),
        #            nn.ReLU(),
        #            nn.Linear(64,3),
                    #nn.Softmax(dim=1),
        #        )
        
        self.lstm = nn.GRU(256,50,bidirectional=True,batch_first=True)        
        self.layers = nn.Linear(100,3)


    def forward(self,text,tokenizer):

        model_input = tokenizer(text,padding=True,return_tensors="pt")
        output = self.bert(**model_input)
    
        #output = output[0][:,0,:]

        #output = output[0]


        _,hidden = self.lstm(output[0])
        final_hidden = torch.cat([hidden[0,:,:],hidden[1,:,:]],dim=1)
    

        out = self.layers(final_hidden)
        return out



    
    def freeze_weights(self):
        for param in self.bert.parameters():
            param.require_grad = False



