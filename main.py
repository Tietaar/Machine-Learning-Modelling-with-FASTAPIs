from fastapi import FastAPI
import uvicorn
import json
from pydantic import BaseModel
import joblib
import json
import imblearn
import pandas as pd
from xgboost import XGBClassifier
from fastapi import FastAPI, Query, Request, HTTPException


app= FastAPI(debug=True)

#load best model using joblip
model= joblib.load("models/Xgboost_model.joblib")

@app.get('/')
def home():
    return{"Message":"Welcome to Sepsis prediction using FastAPI"}
def classify(prediction):
    if prediction == 0:
        return "Patient does not have sepsis"
    else:
        return "Patient has sepsis"
        
@app.post("/predict/")
async def predict_sepsis(
    request: Request,
    prg: float = Query(..., description="Plasma_glucose"),
    pl: float = Query(..., description="Blood_Work_R1"),
    pr: float = Query(..., description="Blood_Pressure"),
    sk: float = Query(..., description="Blood_Work_R2"),
    ts: float = Query(..., description="Blood_Work_R3"),
    m11: float = Query(..., description="BMI"),
    bd2: float = Query(..., description="Blood_Work_R4"),
    ins: int = Query(..., description="Insurance"),
    age: int = Query(..., description="Age")
    # ... (other input parameters)
):
    input_data = [prg, pl, pr, sk, ts, m11, bd2,ins, age]

    input_df = pd.DataFrame([input_data], columns=[
        "Plasma_glucose", "Blood_Work_R1", "Blood_Pressure",
        "Blood_Work_R2", "Blood_Work_R3",
        "BMI", "Blood_Work_R4","Insurance", "Age"
    ])

    pred = model.predict(input_df)
    output = classify(pred[0])

    response = {
        "prediction": output
    }

    return response


if __name__== '__main__':
    uvicorn.run(app)

