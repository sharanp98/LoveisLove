from requests import get
from bs4 import BeautifulSoup
import re
import pandas as pd


url = 'https://www.azquotes.com/quotes/topics/anti-gay.html'
response = get(url)

html_soup = BeautifulSoup(response.text, 'html.parser')

quotes = []

for quote in html_soup.find_all('a', id=re.compile('title_quote_link_.*')):
    quotes.append(quote.text)

sentiment = [0]*len(quotes)

# Create the dataframe
df = pd.DataFrame(list(zip(quotes, sentiment)), columns=['Quote', 'Sentiment'])

df.to_csv('train.csv')
