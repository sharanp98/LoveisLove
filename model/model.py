import torch
import torch.nn as nn
import torch.nn.functional as F

class SentimentModel(nn.Module):

    def __init__(self,bert):
        
        super().__init__()
        self.bert = bert
        self.lstm = nn.GRU(256,50,bidirectional=True,batch_first=True)        
        self.linear = nn.Linear(100,3)


    def forward(self,text,tokenizer):

        model_input = tokenizer(text,padding=True,return_tensors="pt")
        #print(model_input)

        output = self.bert(model_input['input_ids'])
    
        #output = output[0]


        _,hidden = self.lstm(output)
        final_hidden = torch.cat([hidden[0,:,:],hidden[1,:,:]],dim=1)
    

        out = F.softmax(self.linear(final_hidden),dim=1)
        return out



    
    #def freeze_weights(self):
    #    for param in self.bert.parameters():
    #        param.require_grad = False



