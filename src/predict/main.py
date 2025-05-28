from fastapi import FastAPI
import pandas as pd
import mlflow.sklearn
import os

app = FastAPI()
model = mlflow.sklearn.load_model("models/mlflow_model")

@app.get("/")
def read_root():
    return {"message": "Churn API is running"}

@app.post("/predict/")
def predict_churn(data: dict):
    df = pd.DataFrame([data])
    prediction = model.predict(df)[0]
    return {"churn_prediction": bool(prediction)}
