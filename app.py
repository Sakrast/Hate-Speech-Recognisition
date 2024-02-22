from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from fastapi.templating import Jinja2Templates
from starlette.responses import RedirectResponse
from fastapi.responses import Response
import uvicorn
import re
import string


app = FastAPI()

@app.get("/", tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")


model = load_model("model.h5")
with open("tokenizer.pickle", "rb") as handle:
    tokenizer = pickle.load(handle)


class InputData(BaseModel):
    text: str = "     Please enter the text you want to check           "


def clean_text(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text


@app.post('/predict')
async def predict(data: InputData):
    text = clean_text(data.text)
    sequence = tokenizer.texts_to_sequences([text])
    sequence = pad_sequences(sequence, maxlen=300)
    prediction = model.predict(sequence)
    label = "Hate Speech" if prediction > 0.5 else "Not Hate Speech"
    return {'prediction': label}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)

