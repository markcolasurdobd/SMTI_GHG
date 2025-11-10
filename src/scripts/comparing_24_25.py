import pandas as pd
from src.data import PredictData
import src.predicting as pred

cat = 'CAT 11'
cat_24 = 'CAT 11 (USE)'

def exclude_entries(df, col, target, str):
    df.loc[df[col].str.contains(str, case = False), target] = 'EXCLUDE'
    return df

# Load csv
path = r"./data/2025.csv"
df25 = pd.read_csv(path)
df25['CAT 11'] = ''

# Filter Product Category
to_exclude = ['software', 'sw', 'do not use']
for word in to_exclude:
    df25 = exclude_entries(df25, "Product Category", 'CAT 11', word)


# Filter Product Line
to_exclude = ['services', 'fees', 'incentive']
for word in to_exclude:
    df25 = exclude_entries(df25, "Product Line", 'CAT 11', word)

# Filter Product Set
to_exclude = ['do not use', 'tie out', 'service', 'agreement', 'contract', 'shipping', 'handling', 'freight', 'royalty',
             'royalties', 'fee ', 'fees', 'time and materials', 'maintenance', 'software', 'training', 'trade', 'rental',
             'rebate', 'unknown', 'operating', 'labor']
for word in to_exclude:
    df25 = exclude_entries(df25, 'Product Set', 'CAT 11', word)

# Filter Material ID for MMS and PS
df25.loc[(df25['BU'] == 'MMS') & (df25['Material ID'].str.contains('PC')), 'CAT 11'] = 'EXCLUDE'
df25.loc[(df25['BU'] == 'PS') & (df25['Material ID'].str.contains('PC')), 'CAT 11'] = 'EXCLUDE'

# Filter Material Description
to_exclude = ['dummy', 'exclude']
for word in to_exclude:
    df25 = exclude_entries(df25, 'Material Description', 'CAT 11', word)

# Filter Comments that have text
comments = df25[~df25[' COMMENTS '].isna()]
comments_exclude = comments[' COMMENTS '].str.contains('exclude', case = False)
df25.loc[comments_exclude.index, 'CAT 11'] = 'EXCLUDE'

# Material ID column to string
df25['Material ID'] = df25['Material ID'].astype(str)

# Read in 2024 data
df24 = pd.read_csv('./data/2024.csv')

# Drop excluded CAT 11
exclude_rows = df24['CAT 11 (USE)'].str.contains('exclude', case = False)
df24 = df24.drop(df24[exclude_rows].index)

# Drop duplicate Material ID
df24 = df24.drop_duplicates(subset=['Material ID'])

# Left merge on Material ID
df25_filt = df25[df25['CAT 11'] == '']
df25_filt = df25_filt.merge(df24[['Material ID', 'CAT 11 (USE)']], on = 'Material ID', how = 'left')
df25_filt.index = df25[df25['CAT 11'] == ''].index
df25.loc[df25['CAT 11'] == '', 'CAT 11'] = df25_filt['CAT 11 (USE)']
nan25 = df25[df25['CAT 11'].isna()]
df25.loc[(~df25['CAT 11'].isna()) & (df25[' COMMENTS '].isna()), ' COMMENTS '] = 'Matched by 2024 Material ID'

# Classify nan25 based on trained model
d25 = PredictData()
d25.df = nan25
X_columns = ['SEG', 'BU', 'Platform', 'Product Category', 'Product Line', 'Product Set']
d25.X = d25.df[X_columns]
d25.transform()

# Load models for inference
# Load model
model_path = r"C:\Users\10354191\OneDrive - BD\Projects\SMTI\GHG\models\model_24.pkl"
model = pred.load_model(model_path)

# Load vectorizer
vec_path = r"C:\Users\10354191\OneDrive - BD\Projects\SMTI\GHG\models\vec_24.pkl"
vectorizer = pred.load_vectorizer(vec_path)

# Run prediction
preds = pred.predict(model, vectorizer, d25.X)
preds.index = nan25.index
nan25['CAT 11'] = preds.preds
df25.loc[nan25.index, 'CAT 11'] = nan25['CAT 11']
df25.loc[nan25.index, ' COMMENTS '] = 'Classified by ML model'

# Export results
df25.to_csv('./data/2025_results.csv')
