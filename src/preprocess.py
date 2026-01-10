import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def get_data(path="data/churn.csv"):
    df = pd.read_csv(path)
    return df

def preprocess_data(df):
    df = df.drop(["RowNumber", "CustomerId", "Surname"], axis=1)

    X = df.drop("Exited", axis=1)
    y = df["Exited"]

    categorical_features = ["Geography", "Gender"]
    numeric_features = [col for col in X.columns if col not in categorical_features]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("num", StandardScaler(), numeric_features)
        ]
    )

    return X, y, preprocessor
