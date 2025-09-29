# Import libraries
import pandas as pd
import numpy as np

def num_uni_cats(df):
    return {col:len(df[col].unique()) for col in df.columns}

def lower_df(df):
    return df.map(lambda x: x.lower() if isinstance(x, str) else x)

def is_subcat(df, col1, col2):
    df_grouped = df.groupby(col1)[col2].nunique()
    return df_grouped

# Read in data
DATA_PATH = "./data/ghg_filtered.csv"
df = pd.read_csv(DATA_PATH)

# Filter columns with empy rows
na_cols = [col for col in df.columns if df[col].isna().sum() > 0]

#df.to_csv("./data/ghg_filtered.csv", index
subcats = {col:df.groupby(col)['CAT 11 (USE)'].nunique().mean() for col in df.columns}



