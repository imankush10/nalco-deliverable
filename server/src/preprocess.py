import pandas as pd

def preprocess_input(scaler, data, features):
    # Convert the input data into a DataFrame
    df = pd.DataFrame([data])

    # Ensure all required features are present
    for feature in features:
        if feature not in df.columns:
            raise ValueError(f"Missing required feature: {feature}")

    # Arrange features in the correct order
    df = df[features]

    # Scale the data
    return scaler.transform(df)


def preprocess_input_file(scaler,data,features):
    if type(data) != pd.DataFrame:
        df = pd.DataFrame(data)
    else:
        df = data

    for feat in features:
        if feat not in df.columns:
            raise KeyError(f"{feat} key doesn't exist")
    
    df = df[features]

    return scaler.transform(df)
