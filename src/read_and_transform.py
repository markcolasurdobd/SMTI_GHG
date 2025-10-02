import pandas as pd

def training_data(file_path, sheet_name, columns, header):
    # Load data
    print(f"Extracting data from {file_path}")
    df = pd.read_excel(file_path, sheet_name=sheet_name, header=header-1, usecols=columns)

    # Drop any rows with NaN
    df = df.dropna()

    # Cast all dtypes as str
    df = df.astype(str)

    # Lower-case all text
    df = df.map(lambda x: x.lower())

    # Transform any inclusion of 'exclude' to just 'exclude'
    y = df.iloc[:, -1]
    for i, row in enumerate(y):
        if 'exclude' in row:
            y[i] = 'exclude'

    # Turn features into single string
    X = [df.iloc[i, :6].str.cat(sep=' ') for i in range(len(df))]

    # Create output df
    output_df = pd.DataFrame(columns=['feature', 'target'])
    output_df['feature'] = X
    output_df['target'] = y

    # Remove rows containing 'blank'
    blank_rows = []
    for i, row in enumerate(X):
        if 'blank' in row:
            output_df.drop(index=i, inplace=True)

    X = output_df['feature']
    y = output_df['target']

    return X, y

def predict_data(file_path, sheet_name, columns, header):
    print(f"Extracting data from {file_path}")
    # Load data
    df = pd.read_excel(file_path, sheet_name=sheet_name, header=header-1, usecols=columns)

    # Drop any rows with NaN
    df = df.dropna()

    # Cast all dtypes as str
    df = df.astype(str)

    # Lower-case all text
    df = df.map(lambda x: x.lower())

    # Turn features into single string
    X = [df.iloc[i, :6].str.cat(sep=' ') for i in range(len(df))]

    # Create output df
    output_df = pd.DataFrame(columns=['feature'])
    output_df['feature'] = X

    # Remove rows containing 'blank'
    blank_rows = []
    for i, row in enumerate(X):
        if 'blank' in row:
            output_df.drop(index=i, inplace=True)

    X = output_df['feature']

    return X