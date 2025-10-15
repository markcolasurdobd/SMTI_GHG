from src.data import TrainData, ValidationData
import pandas as pd
from sklearn.model_selection import train_test_split

def make_2024():
    # Make the 2024 dataset
    FILE_PATH = r"C:\Users\10354191\OneDrive - BD\Projects\SMTI\GHG\data\GHG FY24 Data for MC.xlsx"
    SHEET_NAME_24 = 'FY 24 EXEMPLAR categorizations'
    COLUMNS_24 = 'A, B, F, H, L, M, Q, X'
    HEADER_24 = 2
    data = TrainData()
    data.load_excel(FILE_PATH, SHEET_NAME_24, HEADER_24, COLUMNS_24)
    data.transform()
    data.remove_value('blank', data.X)
    data.replace_substring('exclude')
    return data

def make_2021():
    # Make the 2021 dataset
    FILE_PATH = r"C:\Users\10354191\OneDrive - BD\Projects\SMTI\GHG\data\GHG FY24 Data for MC.xlsx"
    SHEET_NAME_21 = 'FY21 categorized'
    COLUMNS_21 = 'A, B, C, D, E, G, H, L'
    HEADER_21 = 3
    data = TrainData()
    data.load_excel(FILE_PATH, SHEET_NAME_21, HEADER_21, COLUMNS_21)
    data.transform()
    data.remove_value('erm', data.y)
    data.y.replace({'chloroprep': 'chloraprep', 'bactec machine': 'bactec fx machine'}, inplace=True)
    return data

def make_2023():
    FILE_PATH = "./data/FY23 Cat 11 & 12 data for MC 9 29 25.xlsx"
    SHEET_NAME = ("FY23 Exemplar Categorizations")
    HEADER = 6
    COLUMNS = "B, C, D, E, F, G, L"
    data = ValidationData()
    data.load_excel(FILE_PATH, SHEET_NAME, HEADER, COLUMNS)
    data.transform()
    return data

def make_master(data: list):
    # Pull X and y values from data
    X_list = [item.X for item in data]
    y_list = [item.y for item in data]
    # Vertically concatenate data
    X = pd.concat(X_list, axis = 0)
    y = pd.concat(y_list, axis = 0)
    # Horizontally concatenate X and y
    df = pd.concat([X, y], axis = 1)
    df.to_csv('./data/master.csv', index=False)

def split_master(filename='master.csv'):
    # Load master
    dir = './data'
    path = os.path.join(dir, filename)
    df = pd.read_csv(path)
    # Split X and y
    X = df.iloc[:, 0]
    y = df.iloc[:, -1]
    # Split into train and val_test
    X_train, X_val_test, y_train, y_val_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=0.5, random_state=42)
    # Concatenate back into individual DataFrames
    df_train = pd.concat([X_train, y_train], axis=1)
    df_val = pd.concat([X_val, y_val], axis=1)
    df_test = pd.concat([X_test, y_test], axis=1)
    # Save to data folder
    df_train.to_csv('./data/train.csv', index=False)
    df_val.to_csv('./data/validation.csv', index=False)
    df_test.to_csv('./data/test.csv', index=False)