import pandas as pd
from src.data import PredictData
import src.predicting as pred

# Load data
ref_path = r"./data/2024.csv"
tar_path = r"./data/2025.csv"
ref = pd.read_csv(ref_path)
tar = pd.read_csv(tar_path)

# Add cat 11 and 12 columns to target df
cat11 = 'CAT 11 (USE)'
cat12 = 'CAT 12 (EoL)'

# Label category excluded based on comments
tar.loc[~tar[' COMMENTS '].isna(), [cat11, cat12]] = 'excluded'

# Now, branch off the subset of rows that are not excluded
tar_inc = tar[tar[cat11].isna()]

# Convert Material ID column to string
tar_inc['Material ID'] = tar_inc['Material ID'].astype(str)
ref['Material ID'] = ref['Material ID'].astype(str)
ref.drop_duplicates(subset=['Material ID'], inplace = True)

# Match/merge on Material ID
match = tar_inc.merge(ref[['Material ID', cat11, cat12]], on = 'Material ID', how = 'left')
tar_inc[[cat11, cat12]] = match[[cat11 + '_y', cat12 + '_y']].values

# Now, branch off the remaining, unmatched rows for machine learning
tar_unmatched = tar_inc[tar_inc[cat11].isna()]

# Start here

# Classify unmatched rows based on trained model
d25 = PredictData()
X_columns = ['SEG', 'BU', 'Platform', 'Product Category', 'Product Line', 'Product Set']
d25.df = df25_unmatched
d25.transform()
]
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
df25[cat] = df25[cat].str.lower()

# Export results
df25.to_csv('./results/2025_cat12_results_2.csv')
