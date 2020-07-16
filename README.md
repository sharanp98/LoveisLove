# Love is Love
Classify images uploaded during Pride Month based on its sentiment (positive, negative, or random) and categorize them for internal reference and SEO optimization.
The project is to build an engine that combines the concepts of OCR and NLP that accepts a .jpg file as input, extracts the text, if any, and classifies sentiment as positive or negative. If the text sentiment is neutral or an image file does not have any text, then it is classified as random.

# Steps
## 1. Build an OCR model
1. Preprocessing : the data was preprocessed using the following techniques: Adaptive Thresholding and then Denoising the image. 
2. Extract text from the image : the preprocessed images were passed through pytesseract library 
3. Creating the test dataset : The extracted text was saved in csv(Excel) format with two columns : 
   1. Filename
   2. Detected Text

## 2. Build the training dataset for Natural Language Processing
- The model should categorize the detected text in the following classes:
1. Positive - Quotes in favor of the LGBT community
2. Negative - Quotes against the LGBT community
3. Random - Images that do not have text or do not have anything to do with the LGBT community

- There were not a lot of anti-LGBTQ quotes available on the internet. So, a transfer learning model using BERT was built. 

- Data Scrapper : The data was scrapped from websites using BeautifulSoup. 
   - The scapper was built to extract 50 quotes from each category (Positive, Negative and Random)
   - These quotes were manually labelled as 0 for positive, 1 for negative and 2 for random.
