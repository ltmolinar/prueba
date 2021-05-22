import json
import numpy as np
import os
import pickle
import joblib

def init():
    global model
    # AZUREML_MODEL_DIR is an environment variable created during deployment.
    # It's the path to the model folder (./azureml-models/$MODEL_NAME/$VERSION).
    # For multiple models, it points to the folder containing all deployed models (./azureml-models).
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'model.pkl')
    #model_path = Model.get_model_path(model)
    model = joblib.load(model_path)


def run(raw_data):
    data = np.array(json.loads(raw_data)['data'])
    # Make prediction.
    y_hat = model.predict(data)
    predicted_clases = ['LINEA DE SERVICIO', 'OFICINA DE SERVICIO', 'OFICINA VIRTUAL']
    # You can return any data type as long as it's JSON-serializable.
    # return the result back
    return json.dumps({"predicted_chanel": predicted_clases[int(y_hat)]})