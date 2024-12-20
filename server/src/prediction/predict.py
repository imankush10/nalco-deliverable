from fastapi import HTTPException
import joblib
import pandas as pd

from src.models import load_pickle_model

def preprocess_data(df:pd.DataFrame):
    df = df.dropna()
    df = df.drop_duplicates()
    # remove rows with negative values
    # df = df[(df >= 0).all(1)]
    df = df.reset_index(drop=True)
    return df

def all_value_prediction(data: dict):
    # MODEL_PATH = "models/all3/dlgbm.pkl"
    MODEL_PATH = "models/all3/rf_model_direct_all.pkl"
    SCALER_PATH = "models/all3/scaler.pkl"
    try:
        # features
        FEATURES = ['EMUL_OIL_L_TEMP_PV_VAL0', 'STAND_OIL_L_TEMP_PV_REAL_VAL0', 'GEAR_OIL_L_TEMP_PV_REAL_VAL0', 'EMUL_OIL_L_PR_VAL0', 'ROD_DIA_MM_VAL0', 'QUENCH_CW_FLOW_EXIT_VAL0', 'CAST_WHEEL_RPM_VAL0', 'BAR_TEMP_VAL0', 'QUENCH_CW_FLOW_ENTRY_VAL0', 'GEAR_OIL_L_PR_VAL0', 'STANDS_OIL_L_PR_VAL0', 'TUNDISH_TEMP_VAL0', 'RM_MOTOR_COOL_WATER__VAL0', 'ROLL_MILL_AMPS_VAL0', 'RM_COOL_WATER_FLOW_VAL0', 'EMULSION_LEVEL_ANALO_VAL0', 'furnace_temp', '%SI', '%FE', '%TI', '%V', '%MN', 'OTHIMP', '%AL']

        Y_FEATURES = ['uts', 'conductivity', 'elongation']

        # validate data has all features
        if not all([feature in data.keys() for feature in FEATURES]):
            print("Missing feature in data")
            raise HTTPException(status_code=400, detail="Missing feature in data")

        # load model and scaler
        # model = joblib.load(MODEL_PATH)
        model = load_pickle_model(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)

        # preprocess input
        X_processed = scaler.transform([[data[feature] for feature in FEATURES]])
        y_pred = model.predict(X_processed)

        return dict(zip(Y_FEATURES, y_pred[0]))


    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")


def all_value_prediction_file(data: pd.DataFrame):
    # df = pd.DataFrame(columns=['uts', 'conductivity', 'elongation'])
    uts = []
    conductivity = []
    elongation = []
    # df = []
    for i in data.iterrows():
        a = all_value_prediction(i[1])
        uts.append(a['uts'])
        conductivity.append(a['conductivity'])
        elongation.append(a['elongation'])
    df = pd.DataFrame({'uts': uts, 'conductivity': conductivity, 'elongation': elongation})
    return df


# def all_value_prediction_file(data: pd.DataFrame):
#     # MODEL_PATH = "models/all3/dlgbm.pkl"
#     MODEL_PATH = "models/all3/rf_model_direct_all.pkl"
#     SCALER_PATH = "models/all3/scaler.pkl"
#     print("data is: ", data)
#     try:
#         # features
#         FEATURES = ['EMUL_OIL_L_TEMP_PV_VAL0', 'STAND_OIL_L_TEMP_PV_REAL_VAL0', 'GEAR_OIL_L_TEMP_PV_REAL_VAL0', 'EMUL_OIL_L_PR_VAL0', 'ROD_DIA_MM_VAL0', 'QUENCH_CW_FLOW_EXIT_VAL0', 'CAST_WHEEL_RPM_VAL0', 'BAR_TEMP_VAL0', 'QUENCH_CW_FLOW_ENTRY_VAL0', 'GEAR_OIL_L_PR_VAL0', 'STANDS_OIL_L_PR_VAL0', 'TUNDISH_TEMP_VAL0', 'RM_MOTOR_COOL_WATER__VAL0', 'ROLL_MILL_AMPS_VAL0', 'RM_COOL_WATER_FLOW_VAL0', 'EMULSION_LEVEL_ANALO_VAL0', 'furnace_temp', '%SI', '%FE', '%TI', '%V', '%MN', 'OTHIMP', '%AL']

#         Y_FEATURES = ['uts', 'conductivity', 'elongation']

#         # validate data has all features
#         if not all([feature in data.columns for feature in FEATURES]):
#             print("Missing feature in data")
#             raise HTTPException(status_code=400, detail="Missing feature in data")

#         # print(data)
#         # load model and scaler
#         # model = joblib.load(MODEL_PATH)
#         model = load_pickle_model(MODEL_PATH)
#         scaler = joblib.load(SCALER_PATH)

#         data = preprocess_data(data)
#         print("we are here")
#         # preprocess input
#         X_processed = scaler.transform(data[FEATURES])
#         print(X_processed)
#         y_pred = model.predict(X_processed)

#         print(y_pred)
#         # return dict(zip(Y_FEATURES, y_pred[0]))
#         return pd.DataFrame(y_pred, columns=Y_FEATURES)

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")

