from fastapi import (
    BackgroundTasks,
    FastAPI,
    File,
    UploadFile,
    HTTPException,
    Form,
    Depends,
    Query,
)
from fastapi.responses import FileResponse
from fastapi.responses import JSONResponse
import io
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from pycaret.regression import (
    setup,
    compare_models,
    create_model,
    predict_model,
    plot_model,
    tune_model,
    save_model,
)
from src.prediction.predict import all_value_prediction, all_value_prediction_file
from src.plots.multi_plots import plot_temperature_distribution
from src.reverse_prediction import reverse_features_prediction
from src.models import load_pickle_model
from src.preprocess import preprocess_input, preprocess_input_file
from src.settings import MODEL_PATHS, FEATURES  # Import from settings
import logging
import os
import pandas as pd
from src.plots.plot_utils import SINGLE_PLOTS_PARAMS, call_single_plot
from fastapi.middleware.cors import CORSMiddleware


import numpy as np
from pymodbus.client import ModbusSerialClient
import asyncio
from concurrent.futures import ThreadPoolExecutor
import nest_asyncio
from typing import Dict, List
from src.train import train as tm
from src.sql.dbConnect import engine, SessionLocal, get_db
from sqlalchemy.orm import Session
from pydantic import BaseModel
from datetime import datetime, timedelta
from src.sql.schema import RealTimeData


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load models and scalers
loaded_models = {
    key: {
        "scaler": load_pickle_model(paths["scaler"]),
        "model": load_pickle_model(paths["model"]),
    }
    for key, paths in MODEL_PATHS.items()
}

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

# FastAPI app
app = FastAPI()
executor = ThreadPoolExecutor()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to specific domains if needed
    allow_credentials=True,
    allow_methods=["*"],  # Allow specific HTTP methods if needed
    allow_headers=["*"],  # Specify allowed headers
)

UPLOAD_FOLDER = "uploaded_files"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.post("/upload-predict")
async def upload_and_predict(file: UploadFile = File(...)):
    filename = file.filename
    try:
        # Validate file extension
        if not filename.endswith((".csv", ".xlsx")):
            raise HTTPException(
                status_code=400, detail="File must be a CSV or Excel file"
            )

        # Read the file into a Pandas DataFrame
        if filename.endswith(".csv"):
            df = pd.read_csv(file.file)
        elif filename.endswith(".xlsx"):
            df = pd.read_excel(file.file)

        # Perform prediction
        df = all_value_prediction_file(df)

        # Save the updated file
        output_path = os.path.join(UPLOAD_FOLDER, f"predicted_{filename}")
        if filename.endswith(".csv"):
            df.to_csv(output_path, index=False)
        elif filename.endswith(".xlsx"):
            df.to_excel(output_path, index=False)

        # Peek at the first few rows of the updated file
        preview = df.head(5).to_dict(orient="records")

        return {
            "preview": preview,
            "download_url": f"/download/{os.path.basename(output_path)}",
        }

    except Exception as e:
        logger.error(f"Error processing file: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing file: {e}")


@app.get("/download/{filename}")
def download_file(filename: str):
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(
        filepath, media_type="application/octet-stream", filename=filename
    )


# Input validation
class PredictionRequest(BaseModel):
    property: str  # One of: "uts", "conductivity", or "elongation"
    data: dict


@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        # print(request)
        print(loaded_models.keys())
        # Validate property
        if request.property not in loaded_models.keys():
            return HTTPException(status_code=400, detail="Invalid property type")

        # Select the appropriate model and features
        scaler = loaded_models[request.property]["scaler"]
        model = loaded_models[request.property]["model"]
        features = FEATURES[request.property]  # Get features from settings

        # Preprocess input
        processed_data = preprocess_input(scaler, request.data, features)

        # Make prediction
        predictions = model.predict(processed_data)

        return {"property": request.property, "prediction": predictions.tolist()}

    except ValueError as ve:
        return HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        return HTTPException(status_code=500, detail=f"Error: {str(e)}")


class MultiPredictionRequest(BaseModel):
    data: dict


@app.post("/predict_all")
def predict_all(request: MultiPredictionRequest):
    try:
        # print(all_value_prediction(request.data))
        return all_value_prediction(request.data)
    except Exception as e:
        return {"error": "Failed to predict values", "message": e}


@app.post("/reverse-predict")
def reverse_predict(data: dict):
    return reverse_features_prediction(data)


@app.get("/single-plot-options")
def get_plot_types():
    # return {"plot_types": list(SINGLE_PLOTS.keys())}
    return {"data": SINGLE_PLOTS_PARAMS}


@app.post("/plot")
def plot_model_api(inp: dict):
    plot_type = inp.get("plot_type")
    if plot_type not in SINGLE_PLOTS_PARAMS:
        return JSONResponse(
            content={"error": f"Plot type {plot_type} not found"}, status_code=400
        )

    try:
        # plt =  SINGLE_PLOTS[plot_type]()
        plt = call_single_plot(plot_type, inp["data"])
        # plt.grid(b=True, which='major', color='b', linestyle='-')
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        plt.close()  # Free up memory

        # Return the plot as a streaming response
        return StreamingResponse(buf, media_type="image/png")

        # return StreamingResponse(io.BytesIO(plot), media_type="image/png")
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


class TemperatureDistributionParams(BaseModel):
    length_x: float
    length_y: float
    time_total: float
    dt: float


@app.post("/plot-temperature-distribution")
def multi_plot_temperature_distribution(params: TemperatureDistributionParams):
    return plot_temperature_distribution(
        params.length_x, params.length_y, params.time_total, params.dt
    )


class ModbusParams(BaseModel):
    port: str
    slave_id: int
    start_address: int
    num_registers: int


def preprocess_input(scaler, data: Dict, features: List[str]):
    feature_data = [data[feature] for feature in features]
    scaled_data = scaler.transform(np.array(feature_data).reshape(1, -1))
    return scaled_data


def read_modbus_data(port: str, slave_id: int, address: int, count: int) -> List[int]:
    client = None
    try:
        client = ModbusSerialClient(
            port=port, baudrate=9600, parity="N", stopbits=1, bytesize=8, timeout=1
        )

        if not client.connect():
            raise ConnectionError("Failed to connect to the Modbus slave.")

        response = client.read_holding_registers(
            address=address, count=count, slave=slave_id
        )

        if response.isError():
            raise ValueError(f"Error reading registers: {response}")

        return response.registers

    except Exception as e:
        raise RuntimeError(f"Exception occurred: {str(e)}")
    finally:
        if client:
            client.close()


def run_in_executor(func, *args):
    try:
        # Create a new event loop for the thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = func(*args)
        loop.close()
        return result
    except Exception as e:
        raise RuntimeError(f"Error in executor thread: {str(e)}")


@app.post("/predict-real-time/")
async def predict_modbus_data(params: ModbusParams):
    try:
        # Ensure executor is defined (e.g., ThreadPoolExecutor)
        loop = asyncio.get_event_loop()
        modbus_data = await loop.run_in_executor(
            executor,  # Replace 'executor' with the actual ThreadPoolExecutor instance
            read_modbus_data,
            params.port,
            params.slave_id,
            params.start_address,
            params.num_registers,
        )

        # Convert to dictionary
        FEATURES = [
            "EMUL_OIL_L_TEMP_PV_VAL0",
            "STAND_OIL_L_TEMP_PV_REAL_VAL0",
            "GEAR_OIL_L_TEMP_PV_REAL_VAL0",
            "EMUL_OIL_L_PR_VAL0",
            "ROD_DIA_MM_VAL0",
            "QUENCH_CW_FLOW_EXIT_VAL0",
            "CAST_WHEEL_RPM_VAL0",
            "BAR_TEMP_VAL0",
            "QUENCH_CW_FLOW_ENTRY_VAL0",
            "GEAR_OIL_L_PR_VAL0",
            "STANDS_OIL_L_PR_VAL0",
            "TUNDISH_TEMP_VAL0",
            "RM_MOTOR_COOL_WATER__VAL0",
            "ROLL_MILL_AMPS_VAL0",
            "RM_COOL_WATER_FLOW_VAL0",
            "EMULSION_LEVEL_ANALO_VAL0",
            "furnace_temp",
            "%SI",
            "%FE",
            "%TI",
            "%V",
            "%MN",
            "OTHIMP",
            "%AL",
        ]
        modbus_data_dict = dict(zip(FEATURES, modbus_data))

        # Perform predictions
        print(modbus_data_dict)
        predictions = all_value_prediction(modbus_data_dict)
        print(predictions)

        # Save results to the database
        # db_entry = {
        #     "modbus_data": modbus_data,
        #     "predictions": predictions
        # }  # Define `db_entry` properly based on your database schema
        # # Todo: add all data to db
        # db = get_db()
        # db.add(db_entry)
        # db.commit()

        # Return success response
        return {
            "status": "success",
            "modbus_data": modbus_data,
            "predictions": predictions,
        }

    except ConnectionError as ce:
        raise HTTPException(status_code=503, detail=str(ce))
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


# model training
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    return await tm.upload_file(file)


@app.post("/preprocess")
def preprocess(project_id: str, test_ratio: float, features: list = None):
    return tm.preprocess(project_id, test_ratio, features)


@app.post("/compare_models")
def compare_models_endpoint(project_id: str, target: str, background: BackgroundTasks):
    return tm.compare_models_endpoint(project_id, target, background)


@app.post("/tune_model")
def tune_model_endpoint(project_id: str, target: str, background: BackgroundTasks):
    return tm.tune_model_endpoint(project_id, target, background)


@app.get("/download_model/{project_id}")
def download_model(
    project_id: str,
):
    return tm.download_model(project_id)


@app.get("/status/{project_id}")
async def check_status(project_id: str):
    return tm.read_status(project_id)


@app.post("/dashboard")
def get_dashboard_data(
    minutes: int = Query(
        ..., description="Number of minutes for the trend, e.g., 10, 15, 20"
    ),
    db: Session = Depends(get_db),
):
    try:
        # Validate minutes input
        if minutes <= 0:
            return {"error": "Minutes must be a positive integer"}

        # Calculate the time range
        current_time = datetime.utcnow()
        start_time = current_time - timedelta(minutes=minutes)

        # Query the database for rows within the time range
        data = db.query(RealTimeData).filter(RealTimeData.timestamp >= start_time).all()

        response = {
            "labels": [row.timestamp.strftime("%Y-%m-%d %H:%M:%S") for row in data],
            "datasets": [
                {
                    "label": "UTS",
                    "data": [row.Uts for row in data],
                    "borderColor": "rgba(75, 192, 192, 1)",
                    "fill": False,
                },
                {
                    "label": "Elongation",
                    "data": [row.Elongation for row in data],
                    "borderColor": "rgba(153, 102, 255, 1)",
                    "fill": False,
                },
                {
                    "label": "Conductivity",
                    "data": [row.Conductivity for row in data],
                    "borderColor": "rgba(255, 159, 64, 1)",
                    "fill": False,
                },
            ],
        }

        return response

    except Exception as e:  # Fixed indentation
        return {"error": str(e)}


@app.get("/")
def read_root():
    return {"message": "The API is running!"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
