import pandas as pd
import os

class Data:
    def __init__(self):
        self.file_name = None
        self.sheet_name = None
        self.header = None
        self.columns = None
        self.df = None

    def load_excel(self, file_name: str, sheet_name: str, header: int, columns: str):
        print(f'Loading {sheet_name} from {file_name}')
        self.file_name = file_name
        self.sheet_name = sheet_name
        self.header = header
        self.columns = columns
        self.df = pd.read_excel(self.file_name, sheet_name=self.sheet_name, header=self.header-1, usecols=self.columns, engine='openpyxl')

    def load_csv(self, file_name: str):
        self.file_name = file_name
        self.df = pd.read_csv(self.file_name)

    def transform(self):
        print('Transforming data')
        # Drop missing or NaN values
        self.df = self.df.dropna()
        # Convert all values to strings
        self.df = self.df.astype(str)
        # Lower-case all text
        self.df = self.df.map(lambda x: x.lower())

    def save_to_csv(self, output_dir: str, file_name: str):
        path = os.path.join(output_dir, file_name)
        self.df.to_csv(path, index=False)

class TrainData(Data):
    def __init__(self):
            super().__init__()
            self.X = None
            self.y = None

    def transform(self):
        super().transform()
        self.X = pd.Series([self.df.iloc[i, :-1].str.cat(sep=' ') for i in range(len(self.df))])
        self.y = self.df.iloc[:, -1]
        self.X.index = range(len(self.X))
        self.y.index = range(len(self.y))

    def remove_value(self, value: str, from_col: pd.Series):
        idx_list = []
        for i, row in enumerate(from_col):
            if value in row:
                idx_list.append(i)
        self.X.drop(index = idx_list, inplace=True)
        self.y.drop(index = idx_list, inplace=True)

    def replace_substring(self, substring: str):
        for i, row in enumerate(self.y):
            if substring in row:
                self.y.iloc[i] = substring

    def save_to_csv(self, output_dir: str, file_name: str):
        # Concatenate X and y into df
        path = os.path.join(output_dir, file_name)
        print(f'Saving to {path}')
        df = pd.concat([self.X, self.y])
        df.columns = ['features, target']
        # Save to csv
        df.to_csv(path, index=False)

class ValidationData(Data):
    def __init__(self):
            super().__init__()
            self.X = None
            self.y = None
            self.preds= None

    def transform(self):
        super().transform()
        self.X = pd.Series([self.df.iloc[i, :-1].str.cat(sep=' ') for i in range(len(self.df))])
        self.y = self.df.iloc[:, -1]
        self.X.index = range(len(self.X))
        self.y.index = range(len(self.y))

    def remove_value(self, value: str, from_col: pd.Series):
        idx_list = []
        for i, row in enumerate(from_col):
            if value in row:
                idx_list.append(i)
        self.X.drop(index = idx_list, inplace=True)
        self.y.drop(index = idx_list, inplace=True)

    def replace_substring(self, substring: str):
        for i, row in enumerate(self.y):
            if substring in row:
                self.y.iloc[i] = substring

    def save_to_csv(self, output_dir: str, file_name: str):
        # Concatenate X, y, and preds into df
        path = os.path.join(output_dir, file_name)
        print(f'Saving to {path}')
        if self.preds is None:
            print("Saving without preds")
            df = pd.concat([self.X, self.y])
        else:
            assert len(self.preds) == len(self.X), "X and preds must be same size"
            df = pd.concat([self.X, self.y, self.preds])
        df.columns = ['features, target', 'prediction']
        # Save to csv
        df.to_csv(path, index=False)