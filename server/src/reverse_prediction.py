import pandas as pd
import joblib

def reverse_features_prediction(data: dict):
    FEATURES = ["UTS","Conductivity","Elongation"]
    if not all([feature in data.keys() for feature in FEATURES]):
        return {"error": "missing feature in data valid features: {}".format(data.keys())}
    
    res = reverse_features_prediction_initial(data)
    res.update(data)
    Y_FEATURES = ['furnace_temp','TUNDISH_TEMP_VAL0' ]
    for i in Y_FEATURES:
        res.pop(i)
    res = reverse_features_prediction_final(res)
    return res
    # return reverse_features_prediction_final(df.update(data))

def reverse_features_prediction_initial(data: dict):
    X_FEATURES = ['uts', 'conductivity', 'elongation']
    SCALER_PATH = "models/reverse_prediction/scaler.pkl"
    MODEL_PATH = "models/reverse_prediction/reverse_prediction_rf.pkl"

    Y_FEATURES = ['EMUL_OIL_L_TEMP_PV_VAL0', 'STAND_OIL_L_TEMP_PV_REAL_VAL0', 'GEAR_OIL_L_TEMP_PV_REAL_VAL0', 'EMUL_OIL_L_PR_VAL0', 'ROD_DIA_MM_VAL0', 'QUENCH_CW_FLOW_EXIT_VAL0', 'CAST_WHEEL_RPM_VAL0', 'BAR_TEMP_VAL0', 'QUENCH_CW_FLOW_ENTRY_VAL0', 'GEAR_OIL_L_PR_VAL0', 'STANDS_OIL_L_PR_VAL0', 'TUNDISH_TEMP_VAL0', 'RM_MOTOR_COOL_WATER__VAL0', 'ROLL_MILL_AMPS_VAL0', 'RM_COOL_WATER_FLOW_VAL0', 'EMULSION_LEVEL_ANALO_VAL0', 'furnace_temp', '%SI', '%FE', '%TI', '%V', '%MN', 'OTHIMP', '%AL']
    
    # load models
    try:
        scaler = joblib.load(SCALER_PATH)
        model = joblib.load(MODEL_PATH)
    except Exception as e:
        return {"error": str(e)}
    
    # lowercase all data keys
    data = {k.lower(): v for k, v in data.items()}

    # check if data has all X_FEATURES
    if not all([feature in data.keys() for feature in X_FEATURES]):
        return {"error": "missing feature in data"}
    
    X = [[data[feature] for feature in X_FEATURES]]


    # # scale features
    X_scaled = scaler.transform(X)

    # # predict
    y_pred = model.predict(X_scaled)

    # # create a dictionary of predicted values
    result = dict(zip(Y_FEATURES, y_pred[0]))

    return result
    
def reverse_features_prediction_final(data: dict):
    MODEL_PATH = "models/reverse_prediction/reverse_model2.pkl"
    print("final: data", )
    try:
        model = joblib.load(MODEL_PATH)
    except Exception as e:
        return {"error": str(e)}
    
    # Y_FEATURES = ['furnace_temp','TUNDISH_TEMP_VAL0' ]
    # for i in Y_FEATURES:
    #     data.pop(i)
    data['   UTS'] = data.pop('UTS')
    required_features = ['EMUL_OIL_L_TEMP_PV_VAL0', 'EMUL_OIL_L_PR_VAL0', 'CAST_WHEEL_RPM_VAL0', 'GEAR_OIL_L_PR_VAL0',
       'STANDS_OIL_L_PR_VAL0',
       'RM_MOTOR_COOL_WATER__VAL0',
  '%SI', '%FE', '%TI', '%V', '%MN', 'OTHIMP',
       '%AL', '   UTS', 'Elongation','Conductivity']
    

    df = pd.DataFrame(data, index=[0])
    df = df[required_features]


    y_pred = model.predict(df)

    new_data = dict(zip(['furnace_temp','TUNDISH_TEMP_VAL0' ], y_pred[0]))
    data.update(new_data)

    # X_features
    X_FEATS = ['   UTS', 'Elongation','Conductivity']
    for i in X_FEATS:
        data.pop(i)

    return data