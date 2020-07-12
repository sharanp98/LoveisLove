
from model import SentimentModel
from transformers import AutoTokenizer,AutoModel
import torch


bert_model = AutoModel.from_pretrained('google/bert_uncased_L-4_H-256_A-4')
bert_tokenizer = AutoTokenizer.from_pretrained('google/bert_uncased_L-4_H-256_A-4')
model = SentimentModel(bert_model.embeddings.word_embeddings)
#model.freeze_weights()
model.load_state_dict(torch.load('checkpoint.pth'))
model.eval()

text = [ "happy" ,
        "of course gay men dress well. They didnt spend all that time in the closet doing nothing",
      "Be calm because it is okay to be gay " , "it is very happy to know you are sad", "i am going to be sad",
      "married with pride", "100 years of atlantic stories","we will not be erased","few say being LGBT is negative",
      "Extremism in the defense of liberty is no vice. And moderation in the pursuit of justice is no virtue",
      "All young people, regardless of sexual orientation or identity, deserve a safe and supportive environment in which to achieve their full potential.",
      "Labels are for filing. Labels are for clothing. Labels are not for people.",
      ]

out=model(text,bert_tokenizer)
print(out)

#print(model)
