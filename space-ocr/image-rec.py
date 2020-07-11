import requests
import json
import os
import pandas as pd

columns = ['Filename', 'Text']
df = pd.DataFrame(columns=columns)


def ocr_space_file(filename, overlay=False, api_key='67ef63549a88957', language='eng'):
    payload = {'isOverlayRequired': overlay,
               'apikey': api_key,
               'language': language,
               }
    with open(filename, 'rb') as f:
        r = requests.post('https://api.ocr.space/parse/image',
                          files={filename: f},
                          data=payload,
                          )
    return r.content.decode()


dataset = '../preprocessed_data/'
for file in os.listdir(dataset):
    print(file)
    file_path = dataset+file
    result = ocr_space_file(filename=file_path)
    result = json.loads(result)
    parsed_results = result.get("ParsedResults")[0]
    detected_text = parsed_results.get("ParsedText")
    print(detected_text)
    df.loc[len(df)] = [file, detected_text]

df.to_csv('results.csv')
