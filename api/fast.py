import numpy as np
import os
import io
from fastapi import FastAPI
from tensorflow import keras
from fastapi.middleware.cors import CORSMiddleware
#Our Stuff
from hypopredict.features.chunking import chunkify
import api.utils as utils

app = FastAPI()


path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models/Dan/hypopredict_cnn_model.keras')
app.state.model = keras.models.load_model(path)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


@app.get("/")
def read_root():
    message = "Hello from HypoPredict API!"
    return {
        "message": message,
        "version": "0.2.0",
        "status": "restructured"
    }


@app.post("/predict_from_url")
def predict_from_url(X_test_url: dict):
    """
    Takes a link from Google Drive where we stored a .npy file (X_test)
    Loads the .npy file into a numpy array
    Runs prediction using the loaded local model
    Args:
        X_test_url: dict - dictionary with key 'url' containing the Google Drive link to the .npy file
    Returns:
        X_test shape, prediction results
    """

    url = X_test_url['url']
    #Extract the file ID from the Google Drive URL
    file_id = url.split('/')[-2]
    file = utils.drive_download_bytes(file_id)
    X_test = np.load(io.BytesIO(file))


    # Run prediction
    model = app.state.model
    print("\n", model is not None ,"\n")

    predictions = app.state.model.predict(X_test)

    return {
            "predictions": predictions.tolist(),
            "X_test_shape": X_test.shape
            }
