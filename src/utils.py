import pandas as pd

def create_churn_label(df, cutoff_date, churn_window=30):
    df = df.copy()
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    last_purchase = df.groupby("CustomerID")["InvoiceDate"].max().reset_index()
    last_purchase["days_since_last_purchase"] = (cutoff_date - last_purchase["InvoiceDate"]).dt.days

    last_purchase["churned"] = (last_purchase["days_since_last_purchase"] > churn_window).astype(int)

    return last_purchase[["CustomerID", "churned"]]
