import pickle

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()
model = pickle.load(open("LR_model.model", "rb"))


class Item(BaseModel):
    datos: list


@app.post("/predict")
def predict(item: Item):
    pred = int(model.predict(item.datos)[0])
    return {"prediction": pred}