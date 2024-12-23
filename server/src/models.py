import pickle
import joblib

def load_pickle_model(path):
    try:
        with open(path, "rb") as file:
            return pickle.load(file)
    except Exception as e:
        raise RuntimeError(f"Error loading file {path}: {e}")

    
def load_joblib_model(path):
    try:
        return joblib.load(path)
    except Exception as e:
        raise RuntimeError(f"Error loading file {path}: {e}")