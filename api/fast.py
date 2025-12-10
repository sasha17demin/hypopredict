from fastapi import FastAPI

# Preparing for hte imports once we have a proper structure
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()
#app.state.model = load_model(stage="production")  # Placeholder for actual model loading logic


@app.get("/")
def read_root():
    return {
        "message": "Welcome to the Hypopredict API!"
        }

@app.get("/predict")
def get_predict(input_one: float, input_two: float):
    prediction = input_one + input_two  # Placeholder for actual prediction logic
    return {
        "prediction": f"This is where prediction : {prediction} results will be returned."
        }
    #app.state.model.predict(data)  # Placeholder for actual prediction logic
