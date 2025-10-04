from src.Data import TrainData, ValidationData

def make_2024():
    # 2024 data
    FILE_PATH = './data/GHG FY24 Data for MC.xlsx'
    SHEET_NAME_24 = 'FY 24 EXEMPLAR categorizations'
    COLUMNS_24 = 'A, B, F, H, L, M, Q, X'
    HEADER_24 = 2
    data24 = TrainData()
    data24.load(FILE_PATH, SHEET_NAME_24, HEADER_24, COLUMNS_24)
    data24.transform()
    data24.remove_value('blank', data24.X)
    data24.replace_substring('exclude')
    return data24

def make_2021():
    # 2021 data
    FILE_PATH = './data/GHG FY24 Data for MC.xlsx'
    SHEET_NAME_21 = 'FY21 categorized'
    COLUMNS_21 = 'A, B, C, D, E, G, H, L'
    HEADER_21 = 3
    data = TrainData()
    data.load(FILE_PATH, SHEET_NAME_21, HEADER_21, COLUMNS_21)
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