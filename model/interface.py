
from model import SentimentModel
from transformers import AutoTokenizer,AutoModel



bert_model = AutoModel.from_pretrained('google/bert_uncased_L-4_H-256_A-4')
bert_tokenizer = AutoTokenizer.from_pretrained('google/bert_uncased_L-4_H-256_A-4')
model = SentimentModel(bert_model)

text = [ "i am happy" , "i am going to be sad"]
out=model(text,bert_tokenizer)
print(out)

