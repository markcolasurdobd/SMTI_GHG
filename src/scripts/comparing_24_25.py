import pandas as pd
from src.data import PredictData
import src.predicting as pred

cat = 'CAT 12'
cat_24 = 'CAT 12 (EoL)'

# Load csv
path = r"./data/2025.csv"
df25 = pd.read_csv(path)
df25[cat] = None

# Label category excluded based on comments
df25.loc[~df25[' COMMENTS '].isna(), cat] = 'Excluded'

# Now, branch off the subset of rows that are not excluded
df25_inc = df25[df25[cat].isna()]

# Material ID column to string
df25_inc['Material ID'] = df25_inc['Material ID'].astype(str)

# Read in 2024 data
df24 = pd.read_csv('./data/2024_w_cat12.csv')

# Drop excluded CAT 11
exclude_rows = df24[cat_24].str.contains('exclude', case = False)
df24 = df24.drop(df24[exclude_rows].index)
df24 = df24.drop_duplicates(subset=['Material ID'])

# Match/merge on Material ID
match = df25_inc.merge(df24[['Material ID', 'CAT 12 (EoL)']], on = 'Material ID', how = 'left')
df25_inc[cat] = match[cat_24].values

# Now, branch off the remaining, unmatched rows for machine learning
df25_unmatched = df25_inc[df25_inc[cat].isna()]

# Classify unmatched rows based on trained model
d25 = PredictData()
d25.df = df25_unmatched
X_columns = ['SEG', 'BU', 'Platform', 'Product Category', 'Product Line', 'Product Set']
d25.X = d25.df[X_columns]
d25.transform()

# Load models for inference
# Load model
model_path = r"C:\Users\10354191\OneDrive - BD\Projects\SMTI\GHG\models\model_24_cat12.pkl"
model = pred.load_model(model_path)

# Load vectorizer
vec_path = r"C:\Users\10354191\OneDrive - BD\Projects\SMTI\GHG\models\vec_24_cat12.pkl"
vectorizer = pred.load_vectorizer(vec_path)

# Run prediction
preds = pred.predict(model, vectorizer, d25.X)
df25_unmatched[cat] = preds.preds.values

# Relabel primary df
df25.loc[df25_inc.index, cat] = df25_inc[cat]
df25.loc[df25_inc.index, ' COMMENTS '] = 'Matched by Material ID'
df25.loc[df25_unmatched.index, cat] = df25_unmatched[cat]
df25.loc[df25_unmatched.index, ' COMMENTS '] = 'Classified by ML model'

# Export results
df25.to_csv('./result/2025_cat12_results.csv')
