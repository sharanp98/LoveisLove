
from model import SentimentModel
from transformers import AutoTokenizer,AutoModel
import torch


bert_model = AutoModel.from_pretrained('google/bert_uncased_L-4_H-256_A-4')
bert_tokenizer = AutoTokenizer.from_pretrained('google/bert_uncased_L-4_H-256_A-4')
model = SentimentModel(bert_model)
model.freeze_weights()
model.layers.load_state_dict(torch.load('checkpoint.pth'))
model.eval()

text = [ "happy" ,
        "of course gay men dress well. They didnt spend all that time in the closet doing nothing",
      "Be calm because it is okay to be gay " , "it is very happy to know you are sad", "i am going to be sad"]

out=model(text,bert_tokenizer)
print(out)

#print(model)
