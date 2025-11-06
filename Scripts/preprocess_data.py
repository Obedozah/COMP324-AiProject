import arff
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def preprocess_data(dataset_path, fit_scaler=None, fit_columns=None):
    # Load ARFF file and convert to DataFrame
    with open(dataset_path, 'r') as f:
        arff_file = arff.load(f)
    df = pd.DataFrame(arff_file['data'], columns=[attr[0] for attr in arff_file['attributes']])

    # Encode categorical columns
    obj_cols = df.select_dtypes(include=['object']).columns
    df_encoded = pd.get_dummies(df, columns=obj_cols)

    print(df.tail())
    print(df.select_dtypes(include=['object']).tail())
    print(df_encoded.tail())

    # Align test columns to training columns
    if fit_columns is not None:
        df_encoded = df_encoded.reindex(columns=fit_columns, fill_value=0)
        #print("FIT COL\n",df_encoded.head())
    #print("NOT FIT COL\n",df_encoded.head())

    # Scale features
    if fit_scaler is None:
        scaler = StandardScaler()
        df_scaled = scaler.fit_transform(df_encoded)
        print("NONE:\n")
        print(df_scaled)
    else:
        df_scaled = fit_scaler.transform(df_encoded)
        print("SOME:\n")
        print(df_scaled)
        scaler = fit_scaler

    # Fix target column (last column)
    df_scaled[:, -1] = np.where(np.isclose(df_scaled[:, -1], 0.93442518), 0, 1)

    return df_scaled, scaler, df_encoded.columns
