import joblib
import numpy as np
from fastapi import FastAPI

from api.model import Transaction

app = FastAPI()
model = joblib.load("../models/fraud_model.pkl")



@app.post("/predict")
def predict(data: Transaction):
    input_data = np.array([[data.V1, data.V2, ..., data.Amount]])
    prediction = model.predict(input_data)
    return {"fraud": bool(prediction[0])}
