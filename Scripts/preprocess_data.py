import arff
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def preprocess_data(dataset_path, fit_scaler=None, fit_columns=None):
    # Load ARFF file and convert to DataFrame
    with open(dataset_path, 'r') as f:
        arff_file = arff.load(f)
    df = pd.DataFrame(arff_file['data'], columns=[attr[0] for attr in arff_file['attributes']])

    # Separate features and target
    X = df.iloc[:, :-1]
    Y = df.iloc[:, -1]

    print(df.head())
    # Encode categorical columns: Change words to numbers
    obj_cols = X.select_dtypes(include=['object']).columns
    X_encoded = pd.get_dummies(X, columns=obj_cols)

    # Align test columns to column amount from training
    if fit_columns is not None:
        X_encoded = X_encoded.reindex(columns=fit_columns, fill_value=0)

    # Scale features with the scaler from training
    if fit_scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_encoded)
    else:
        X_scaled = fit_scaler.transform(X_encoded)
        scaler = fit_scaler

    # convert Y to 0 or 1
    Y = np.where(Y == 'normal', 0, 1) if Y.dtype == object else Y

    # Combine X and Y
    df_scaled = np.column_stack((X_scaled, Y))

    return df_scaled, scaler, X_encoded.columns