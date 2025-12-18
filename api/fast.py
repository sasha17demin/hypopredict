import numpy as np
import os
import io
from fastapi import FastAPI
from tensorflow import keras
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
#Our Stuff
from hypopredict.chunker import hello
import api.utils as utils
import pickle
import pandas as pd
from hypopredict.daniel_model.lstmcnn import Lstmcnnmodel


class PredictRequest(BaseModel):
    url: str

app = FastAPI()


cnn_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '/home/danielfarkas/code/sasha17demin/hypopredict/hypopredict/daniel_model/checkpoints/baseline/lstmcnn_baseline.keras')
app.state.model_cnn = keras.models.load_model(cnn_path, custom_objects= {"loss": Lstmcnnmodel.focal_loss(alpha=0.75, gamma=2)})

models_XSK_path = '/home/danielfarkas/code/sasha17demin/hypopredict/fusion_model_dict_0_288.pkl'
with open(models_XSK_path, 'rb') as f:
    app.state.models_dict = pickle.load(f)

with open('/home/danielfarkas/code/sasha17demin/hypopredict/hypopredict/daniel_model/results/cnn_indexed_preds_83.pkl', 'rb') as f:
    app.state.indexed_preds_83 = pickle.load(f)

with open('/home/danielfarkas/code/sasha17demin/hypopredict/hypopredict/daniel_model/results/cnn_indexed_preds_64.pkl', 'rb') as f:
    app.state.indexed_preds_64 = pickle.load(f)




app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


@app.get("/")
def read_root():
    message = hello()
    return {
        "message": message
        }


@app.post("/predict_from_url")
def predict_from_url(request : PredictRequest):
    """
    Takes a link from Google Drive where we stored a .npy file (X_test)
    Loads the .npy file into a numpy array
    Runs prediction using the loaded local model
    Args:
        X_test_url: dict - dictionary with key 'url' containing the Google Drive link to the .npy file
    Returns:
        X_test shape, prediction results
    """

    url = request.url
    #Extract the file ID from the Google Drive URL
    file_id = url.split('/')[-2]
    file = utils.drive_download_bytes(file_id)
    X_test = np.load(io.BytesIO(file))




    # Run prediction
    model_cnn = app.state.model_cnn
    print("\n", model_cnn is not None ,"\n")

    predictions = model_cnn.predict(X_test)

    return {
            "predictions": predictions.tolist(),
            "X_test_shape": X_test.shape
            }


@app.get("/predict_fusion_local_64")
def predict_fusion_local_64():

    with open('/home/danielfarkas/code/sasha17demin/hypopredict/notebooks/demo_day_64_seq_demo.pkl', 'rb') as f:
        X_test_cnn = pickle.load(f)

    with open('/home/danielfarkas/code/sasha17demin/hypopredict/64_ml_prepped_split_20251217_183629.pkl', 'rb') as f:
        X_test_models = pickle.load(f)['splits_prepped'][0][0]


    # Run predictions
    X_test_fusion = pd.DataFrame()
    for name, model in app.state.models_dict.items():
       if name == 'fusion':
           pass
       else:
           X_test_fusion[name + '_prob'] = model.predict_proba(X_test_models)[:, 1]

    predictions_cnn = app.state.model_cnn.predict(X_test_cnn)

    y_probs_test_fusion = app.state.models_dict['fusion'].predict_proba(X_test_fusion)[:, 1]


    return {
            "predictions_cnn": predictions_cnn.tolist(),
            "predictions_fusion": y_probs_test_fusion.tolist(),
            "predictions_xgb": X_test_fusion['xgb_prob'].tolist(),
            "predictions_svm": X_test_fusion['svm_prob'].tolist(),
            "predictions_knn": X_test_fusion['knn_prob'].tolist()
            }


@app.post("/predict_fusion_url")
def predict_hybrid_url(request1 : PredictRequest, request2 : PredictRequest):

    with open('/home/danielfarkas/code/sasha17demin/hypopredict/notebooks/demo_day_64_seq_demo.pkl', 'rb') as f:
        X_test_cnn = pickle.load(f)

    with open('/home/danielfarkas/code/sasha17demin/hypopredict/64_ml_prepped_split_20251217_183629.pkl', 'rb') as f:
        splits_prepped = pickle.load(f)['splits_prepped']

    X_demo = splits_prepped[0][0]

    print('Hi')
    print(X_demo)

    url_cnn = request1.url
    #Extract the file ID from the Google Drive URL
    cnn_file_id = url_cnn.split('/')[-2]
    cnn_file = utils.drive_download_bytes(cnn_file_id)
    X_test_cnn = pickle.load(io.BytesIO(cnn_file))

    fusion_url = request2.url
    #Extract the file ID from the Google Drive URL
    fusion_file_id = fusion_url.split('/')[-2]
    fusion_file = utils.drive_download_bytes(fusion_file_id)
    X_test_models = pickle.load(io.BytesIO(fusion_file))


    # Run predictions
    X_test_fusion = pd.DataFrame()
    for name, model in app.state.models_dict.items():
       if name == 'fusion':
           pass
       else:
           X_test_fusion[name + '_prob'] = model.predict_proba(X_test_models)[:, 1]

    predictions_cnn = app.state.model_cnn.predict(X_test_cnn)

    y_probs_test_fusion = app.state.models_dict['fusion'].predict_proba(X_test_fusion)[:, 1]


    return {
            "predictions_cnn": predictions_cnn.tolist(),
            "predictions_fusion": y_probs_test_fusion.tolist(),
            "predictions_xgb": X_test_fusion['xgb_prob'].tolist(),
            "predictions_svm": X_test_fusion['svm_prob'].tolist(),
            "predictions_knn": X_test_fusion['knn_prob'].tolist()
            }
