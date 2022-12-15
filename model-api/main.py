# imports required for running model
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import pandas as pd

from nltk.corpus import stopwords
import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence

import re
import nltk
nltk.download('stopwords')

# imports for creating API

# model functions :----
load_model = keras.models.load_model("./hate&speech_lstm2.h5")

REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))

# text cleaning function


def clean_text(text):
    text = text.lower()
    text = REPLACE_BY_SPACE_RE.sub(' ', text)
    text = BAD_SYMBOLS_RE.sub('', text)
    text = text.replace('x', '')
    text = ' '.join(word for word in text.split() if word not in STOPWORDS)
    return text


# creating tokenizer
df_twitter = pd.read_csv('train.csv')
df_twitter.drop('id', axis=1, inplace=True)
df_offensive = pd.read_csv('labeled_data.csv')
df_offensive.drop(['Unnamed: 0', 'count', 'hate_speech',
                   'offensive_language', 'neither'], axis=1, inplace=True)
df_offensive["class"].replace({0: 1}, inplace=True)
df_offensive.rename(columns={'class': 'label'}, inplace=True)

frame = [df_twitter, df_offensive]
df2 = pd.concat(frame)
df2['tweet'] = df2['tweet'].apply(clean_text)
MAX_NB_WORDS = 50000
MAX_SEQUENCE_LENGTH = 250
EMBEDDING_DIM = 100
tokenizer2 = Tokenizer(
    num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
tokenizer2.fit_on_texts(df2['tweet'].values)


def predict(inp_text):

    print(inp_text)
    test = inp_text
    test = [clean_text(test)]
    seq = tokenizer2.texts_to_sequences(test)
    padded = sequence.pad_sequences(seq, maxlen=250)
    pred = load_model.predict(padded)
    print('pred: ', pred[0][0])
    result = np.array(pred[0][0])
    result = result.tolist()
    resu=[result]
    if pred < 0.4:
        res_text = "no hate"
    else:
        res_text = "hate and abusive"
    resu.append(res_text)
    return resu


# API begins here:---
app = FastAPI()


class Item(BaseModel):
    inp_text: str


class Prediction(BaseModel):
    prediction: str
    score: float


origins = [
    "http://localhost",
    "http://localhost:8080",
    "http://127.0.0.1:5500"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get('/')
async def index():
    return {"message": "Prediction API"}


@app.post('/predict/')
async def predict_review(item: Item):
    res = predict(item.inp_text)
    print(res)
    return {
        "score": res[0],
        "prediction": res[1]
    }


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)