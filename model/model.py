import torch
import torch.nn as nn


class SentimentModel(nn.Module):

    def __init__(self,bert):
        
        super().__init__()
        self.bert = bert
       
        

        


    def forward(self,text,tokenizer):

        model_input = tokenizer(text,padding=True,return_tensors="pt")
        output = self.bert(**model_input)
        output = output[0][:,0,:]
        #print(output.shape)
        final_out = self.layers(output)
        return final_out


    def freeze_weights(self):
        for param in self.bert.parameters():
            param.require_grad = False



