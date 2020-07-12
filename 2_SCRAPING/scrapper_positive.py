from requests import get
from bs4 import BeautifulSoup
import re
import pandas as pd

'''Scrape page 1'''
url = 'https://www.brainyquote.com/topics/gay-quotes'
response = get(url)

html_soup = BeautifulSoup(response.text, 'html.parser')

quotes = []

for quote in html_soup.find_all('a', title='view quote'):
    quotes.append(quote.text)

'''Scrape page 2'''

url = 'https://www.brainyquote.com/topics/gay-quotes_2'
response = get(url)

html_soup = BeautifulSoup(response.text, 'html.parser')

for quote in html_soup.find_all('a', title='view quote'):
    quotes.append(quote.text)

sentiment = [1]*len(quotes)

# Create the dataframe
df = pd.DataFrame(list(zip(quotes, sentiment)), columns=['Quote', 'Sentiment'])

df.to_csv('train_positive.csv')

print(df)
