from requests import get
from bs4 import BeautifulSoup
import re
import pandas as pd


url = 'https://www.wow4u.com/ability2/'
response = get(url)

html_soup = BeautifulSoup(response.text, 'html.parser')

quotes = []

for quote in html_soup.find_all('li', cwidth="661"):
    quotes.append(quote.text.replace("\n", "").replace(",", ""))

sentiment = [2]*len(quotes)

# Create the dataframe
df = pd.DataFrame(list(zip(quotes, sentiment)), columns=['Quote', 'Sentiment'])

df.to_csv('train_neutral.csv')
# print(df)
# print(len(df))
