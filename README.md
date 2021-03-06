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

## 3. Deep Learning Model
- The Deep Learning Model uses a transfer learning approach in which the pretrained model used is [BERT](https://arxiv.org/abs/1810.04805)
- We have used the pretrained BERT model from [transformers](https://huggingface.co/transformers/) having 4 layers with a hidden dimension size of 256.
- Model Architecture
  
  <img src="https://i.imgur.com/lsrPmdH.png" alt="drawing" style="width:200px;height:200px;"/>
  
- Parameters of the BERT model are fixed while the parameters of the RNN and the Linear layer can be trained.
- Model can be found at 3_Model directory
   - model.py  = Model Architecture
   - train.py  = Code for training the model using a training data provided as csv 
   - interface.py = Code for performing inference using the trained model


 ## References 
   - http://jalammar.github.io/a-visual-guide-to-using-bert-for-the-first-time/
   
This project was done as a part of HackerEarth Machine Learning challenge: [Love is love](https://www.hackerearth.com/challenges/competitive/hackerearth-machine-learning-challenge-pride-month-edition/problems/)
