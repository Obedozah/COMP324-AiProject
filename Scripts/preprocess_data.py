import arff
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def preprocess_data(dataset_path):
    # Loading ARFF file and converting to DataFrame
    with open(dataset_path, 'r') as f:
        arff_file = arff.load(f)
    df = pd.DataFrame(arff_file['data'], columns=[attr[0] for attr in arff_file['attributes']])

    # Encoding categorical columns into numerical data
    obj_cols = df.select_dtypes(include=['object']).columns
    df_encoded = pd.get_dummies(df, columns=obj_cols)

    # Scaling numerical data
    scaler = StandardScaler()
    df_preprocessed = scaler.fit_transform(df_encoded)

    # Fixing target
    df_preprocessed[:, -1] = np.where(np.isclose(df_preprocessed[:, -1], 0.93442518), 0, 1)

    return df_preprocessed