import os
import mlflow

def config_mlflow():
    MLFLOW_TRACKING_URI = 'https://dagshub.com/mariifaria/projeto_integrado_ml'
    MLFLOW_TRACKING_USERNAME = 'mariifaria'
    MLFLOW_TRACKING_PASSWORD = '3cbb9994863f11c83c600986df7d3fff17bae014'
    os.environ['MLFLOW_TRACKING_USERNAME'] = MLFLOW_TRACKING_USERNAME
    os.environ['MLFLOW_TRACKING_PASSWORD'] = MLFLOW_TRACKING_PASSWORD
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    mlflow.tensorflow.autolog(log_models=True, log_input_examples=True, log_model_signatures=True)