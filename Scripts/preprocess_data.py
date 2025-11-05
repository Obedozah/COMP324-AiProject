import arff
import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_data():
    # Loading ARFF file and converting to DataFrame
    with open('datasets/KDDTrain+_20Percent.arff', 'r') as f:
        arff_file = arff.load(f)
    df = pd.DataFrame(arff_file['data'], columns=[attr[0] for attr in arff_file['attributes']])

    # One-hot encoding for service column
    df = pd.get_dummies(df, columns=['service'])

    # Encoding categorical columns into numerical data
    obj_cols = df.select_dtypes(include=['object']).columns
    df_encoded = pd.get_dummies(df, columns=obj_cols, drop_first=True)

    # Scaling numerical data
    scaler = StandardScaler()
    df_preprocessed = scaler.fit_transform(df_encoded)
    return df_preprocessed

df = preprocess_data()
print(df[:5])