import pickle
import joblib
from tensorflow.keras.models import load_model

def load_pickle_model(path):
    try:
        with open(path, "rb") as file:
            return pickle.load(file)
    except Exception as e:
        raise RuntimeError(f"Error loading file {path}: {e}")

def load_tensorflow_model(model_path):
    try:
        model = load_model(model_path)
        return model
    except Exception as e:
        raise RuntimeError(f"Error loading file {model_path}: {e}")
    
def load_joblib_model(path):
    try:
        return joblib.load(path)
    except Exception as e:
        raise RuntimeError(f"Error loading file {path}: {e}")