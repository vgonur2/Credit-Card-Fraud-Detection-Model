import sqlite3
import pandas as pd
import numpy as np
import joblib
from scipy import sparse

#read df
df = pd.read_csv("C:/Users/sunsu/OneDrive/Desktop/Winter ML Project 25-26/loan pred ipynb new/data/processed/output.csv")

#Create SQLite Database
conn = sqlite3.connect("mock_database_bank.db")
cur = conn.cursor()

#drop table if exists alr
cur.execute("DROP TABLE IF EXISTS transactions")

#Create Table automatically using pandas
df.to_sql("transactions", conn, index = False, chunksize=10_000)
df.shape[0]

#Load or .pkl files
model = joblib.load("C:/Users/sunsu/OneDrive/Desktop/Winter ML Project 25-26/loan pred ipynb new/notebooks/fraud_model.pkl")
preprocessor = joblib.load("C:/Users/sunsu/OneDrive/Desktop/Winter ML Project 25-26/loan pred ipynb new/notebooks/preprocessor.pkl")

#Define our Thresholds
def fraud_decision(prob):
    if prob >= 0.90:
        return "AUTO REJECT"
    elif prob >= 0.70:
        return "MANUAL REVIEW"
    else:
        return "AUTO APPROVE"

#Simulating batch scoring from SQL
batch_Size = 10
offset = 0

while True:
    query = f"SELECT * FROM transactions LIMIT {batch_Size} OFFSET {offset}"
    batch_df = pd.read_sql(query,conn)
    if batch_df.empty:
        break
    transaction_ids = batch_df["TransactionID"].values
    features = batch_df.drop(columns=[ 'IsFraud'], errors = 'ignore')

    #Apply some preprocessing as Training
    num_cols = preprocessor["numerical_cols"]
    cat_cols = preprocessor["categorical_cols"]

    X_num = features[num_cols].copy()
    X_cat = features[cat_cols].copy()

    # Apply imputers
    X_num = preprocessor["num_imputer"].transform(X_num)
    X_cat = preprocessor["cat_imputer"].transform(X_cat)

    # Apply encoder
    X_cat = X_cat[preprocessor["encoder"].feature_names_in_]
    X_cat = preprocessor["encoder"].transform(X_cat)

    # Combine numeric and categorical
    if sparse.issparse(X_cat):
        from scipy.sparse import hstack

        X_processed = hstack([X_num, X_cat])
    else:
        X_processed = np.hstack([X_num, X_cat])

    # ----------------------
    # Predict probabilities
    # ----------------------
    y_prob = model.predict(X_processed, num_iteration=model.best_iteration)

    #Apply Decision Thresholds
    df_results = pd.DataFrame({
        "TransactionID": transaction_ids,
        'Fraud_probability':y_prob,
        "Fraud_decision":[fraud_decision(p) for p in y_prob]
    })

    df_results.to_sql("transaction_results", conn, if_exists="append", index=False)

    offset += batch_Size

conn.close()
