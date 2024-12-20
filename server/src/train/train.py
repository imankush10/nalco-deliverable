
import json
from fastapi import BackgroundTasks, FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
import os
import uuid
import pandas as pd
from sklearn.model_selection import train_test_split
from pycaret.regression import setup, compare_models, tune_model, finalize_model, save_model, load_model, create_model


PROJECTS_FOLDER = "./projects"
os.makedirs(PROJECTS_FOLDER, exist_ok=True)

# app = FastAPI()

def parseStepId(step):
    # 1 and increment => upload, preprocess, compare_models, tune_model, finalize_model, download_model
    if step == 'upload':
        return 1
    elif step == 'preprocess':
        return 2
    elif step == 'compare_models':
        return 3
    elif step == 'tune_model':
        return 4
    elif step == 'download_model':
        return 5

# Helper: Update status file
def update_status(project_id,step, status, message=None,error=None):
    status_path = os.path.join(PROJECTS_FOLDER, project_id, "status.json")
    data = {
        "stepId":parseStepId(step),
        "step":step,
        "status":status,
        "message":message,
        "hasError":False if error is None else True,
        "error":error
    }
    with open(status_path, "w") as f:
        json.dump(data, f)

# Helper: Read status file
def read_status(project_id):
    status_path = os.path.join(PROJECTS_FOLDER, project_id, "status.json")
    if os.path.exists(status_path):
        with open(status_path, "r") as f:
            return json.load(f)
    return {"status": "not_found", "message": "No status available"}

def preprocess_data(df:pd.DataFrame):
    df = df.dropna()
    df = df.drop_duplicates()
    # remove rows with negative values
    # df = df[(df >= 0).all(1)]
    df = df.reset_index(drop=True)
    return df

# 1. Upload File & Create Project
# @app.post("/upload")
async def upload_file(file: UploadFile):
    project_id = str(uuid.uuid4())
    project_path = os.path.join(PROJECTS_FOLDER, project_id)
    os.makedirs(project_path, exist_ok=True)
    filename = file.filename
    try:
        # Validate file extension
        if not filename.endswith(('.csv', '.xlsx')):
            raise HTTPException(status_code=400, detail="File must be a CSV or Excel file")

        # Read the file into a Pandas DataFrame
        if filename.endswith('.csv'):
            df = pd.read_csv(file.file)
        elif filename.endswith('.xlsx'):
            df = pd.read_excel(file.file)

        df = preprocess_data(df)

        file_path = os.path.join(project_path, "data.csv")
        df.to_csv(file_path,index=False)
        features = df.columns.tolist()
        update_status(project_id,"upload", "completed", "File uploaded successfully")
        return {"message": "Project created", "project_id": project_id,"features":features}
    except Exception as e:
        return {"message": "Error uploading file", "error": str(e)}

# 2. Preprocess Data
# @app.post("/preprocess")
def preprocess(project_id: str, test_ratio: float, features:list=None):
    project_path = os.path.join(PROJECTS_FOLDER, project_id)
    if not os.path.exists(project_path):
        raise HTTPException(status_code=404, detail="Project not found")
    
    # Locate the uploaded file
    file_path = os.path.join(project_path, "data.csv")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Data file not found")

    # Split the dataset
    df = pd.read_csv(file_path)

    if features is None or features == []:
        feats = []
        for f in features:
            if f in df.columns:
                feats.append(f)
        df = df[feats]

    train, test = train_test_split(df, test_size=test_ratio, random_state=42)
    # test, val = train_test_split(temp, test_size=val_ratio / (test_ratio + val_ratio), random_state=42)

    # Save splits
    train.to_csv(os.path.join(project_path, "train.csv"), index=False)
    test.to_csv(os.path.join(project_path, "test.csv"), index=False)
    # val.to_csv(os.path.join(project_path, "val.csv"), index=False)
    
    update_status(project_id,"preprocess", "completed", "Data preprocessing completed")
    return {"message": "Data preprocessed", "project_id": project_id}


def __compare_models__(project_id:str, data:pd.DataFrame, target:str):
    project_path = os.path.join(PROJECTS_FOLDER, project_id)

    update_status(project_id,"compare_models","started","Process started Successfully")

    setup(data=data, target=target, session_id=42)
    update_status(project_id,"compare_models","in-progress","Process in progress")
    best_model = compare_models(fold=5, turbo=True)

    # print(f"Best model: {best_model}")
    if best_model is None:
        raise HTTPException(status_code=400, detail="No models could be compared")
    
    os.makedirs(project_path, exist_ok=True)
    save_model(best_model, os.path.join(project_path, "best_model"))
    update_status(project_id,"compare_models","completed",f"Best model: {best_model}")


# @app.post("/compare_models")
def compare_models_endpoint(project_id: str, target: str,background:BackgroundTasks):
    print("compare")
    status = read_status(project_id)
    print("status of compare")
    if status["status"] == "not_found":
        raise HTTPException(status_code=404, detail="Project not found")
    elif status["stepId"] < 2:
        return {"message":"Data not preprocessed"}
    
    print("compare model")

    project_path = os.path.join(PROJECTS_FOLDER, project_id)
    train_path = os.path.join(project_path, "train.csv")
    if not os.path.exists(train_path):
        raise HTTPException(status_code=404, detail="Training data not found")

    train_data = pd.read_csv(train_path)
    # print(f"Train data columns: {train_data.columns.tolist()}")
    if target not in train_data.columns:
        raise HTTPException(status_code=400, detail=f"Target column '{target}' not found in training data")
    if train_data.empty:
        raise HTTPException(status_code=400, detail="Training data is empty")
    
    print("compare model")
    # compare model background task
    background.add_task(__compare_models__,project_id,train_data,target)
    
    return {"message": "Model comparison process started"}


def __tune_model__(project_id:str,data:pd.DataFrame,best_model_path:str,  target:str,):
    update_status(project_id,"tune_model","started","Model fine-tuning started successfully")
    project_path = os.path.join(PROJECTS_FOLDER, project_id)
    

    setup(data=data, target=target, session_id=42)
    update_status(project_id,"tune_model","in-progress","Model fine-tune in process")
    best_model = load_model(best_model_path.split(".pkl")[0])

    tuned_model = tune_model(best_model)
    model = create_model(tuned_model)
    save_model(model, os.path.join(project_path, "final_model"))
    update_status(project_id,"tune_model","completed","Fine-tuned model saved successfully")


# 4. Tune Model
# @app.post("/tune_model")
def tune_model_endpoint(project_id: str, target: str,background:BackgroundTasks):
    status = read_status(project_id)
    if status["status"] == "not_found":
        raise HTTPException(status_code=404, detail="Project not found")
    elif status["stepId"] < 3:
        return {"message":"Best model not compared yet"}
    
    project_path = os.path.join(PROJECTS_FOLDER, project_id)
    best_model_path = os.path.join(project_path, "best_model.pkl")
    if not os.path.exists(best_model_path):
        raise HTTPException(status_code=404, detail="No model to tune")
    
    train_path = os.path.join(project_path, "train.csv")
    # print(f"Checking path: {train_path}")
    if not os.path.exists(train_path):
        raise HTTPException(status_code=404, detail="Training data not found")    
    train_data = pd.read_csv(train_path)

    # model tune background task
    background.add_task(__tune_model__,project_id,train_data,best_model_path,target)
    
    return {"message": "Model tuning started"}

# 7. Download Model
# @app.get("/download_model/{project_id}")
def download_model(project_id: str):
    status = read_status(project_id)
    if status["status"] == "not_found":
        raise HTTPException(status_code=404, detail="Project not found")
    elif status["stepId"] < 3:
        return {"message":"Model not trained yet"}
    
    project_path = os.path.join(PROJECTS_FOLDER, project_id)
    final_model_path = os.path.join(project_path,"best_model.pkl" if status['stepId'] == 3 else "final_model.pkl")

    if not os.path.exists(final_model_path):
        raise HTTPException(status_code=404, detail="Model not found")

    return FileResponse(final_model_path, media_type="application/octet-stream", filename="final_model.pkl")

# 5. Check Task Status
# @app.get("/status/{project_id}")
async def check_status(project_id: str):
    return read_status(project_id)