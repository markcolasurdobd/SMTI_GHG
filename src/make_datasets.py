from src.data import TrainData, ValidationData
import pandas as pd
from sklearn.model_selection import train_test_split

def make_2024():
    # Make the 2024 dataset
    FILE_PATH = r"C:\Users\10354191\OneDrive - BD\Projects\SMTI\GHG\data\GHG FY24 Data for MC.xlsx"
    SHEET_NAME_24 = 'FY 24 EXEMPLAR categorizations'
    COLUMNS_24 = 'A, B, F, H, L, M, Q, X'
    HEADER_24 = 2

    data24 = TrainData()
    data24.load_excel(FILE_PATH, SHEET_NAME_24, HEADER_24, COLUMNS_24)
    data24.transform()
    data24.remove_value('blank', data24.X)
    data24.replace_substring('exclude')
    return data24

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

if __name__ == '__main__':
    # Make datasets
    d24 = make_2024()
    d21 = make_2021()

    # Create master.csv
    X = pd.concat([d21.X, d24.X], axis = 0)
    y = pd.concat([d21.y, d24.y], axis = 0)
    df = pd.concat([X, y], axis = 1)
    df.to_csv('./data/master.csv', index = False)

    # Split data
    X_train, X_val_test, y_train, y_val_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    df_train = pd.concat([X_train, y_train], axis = 1)
    df_train.to_csv('./data/train.csv', index = False)
    X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=0.5, random_state=42,
                                                    stratify=y_val_test)
    df_val = pd.concat([X_val, y_val], axis = 1)
    df_test = pd.concat([X_test, y_test], axis=1)

    df_val.to_csv('./data/validation.csv', index = False)
    df_test.to_csv('./data/test.csv', index = False)

