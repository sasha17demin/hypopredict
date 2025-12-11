from fastapi import FastAPI
from hypopredict.train_test_split import hello
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc
import os


# Preparing for hte imports once we have a proper structure
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()
path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models/Dan/hypopredict_cnn_model.keras')
print("\n" ,path, "\n")

app.state.model = keras.models.load_model(path)


@app.get("/")
def read_root():
    message = hello()
    return {
        "message": message
        }

@app.get("/predict")
def get_predict(X_test, y_test):
    """
    At the moment we prepadd X_test and then make prediction.
    In the future, we will have proper input data schema and preprocessing steps.
    Args:
        X_test: input data for prediction [padded]
        y_test: true labels for evaluation
    Returns:
        prediction results

    We will return precision-recall AUC and histogram of predictions.
    We return the histogram as an object to be plotted on the frontend.
    """


    #X_test_preprocess = my_prepprocessing_function(X_test)  # Placeholder for actual preprocessing function
    prediction = app.state.model.predict(X_test)  # Placeholder for actual prediction logic
    precision, recall, _ = precision_recall_curve(y_test, prediction)
    pr_auc = auc(recall, precision)
    return {
        "pr_auc": pr_auc,
        "predictions" : prediction
        }
    #app.state.model.predict(data)  # Placeholder for actual prediction logic
