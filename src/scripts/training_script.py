import pandas as pd
from src.data import TrainData
import src.training as tr

# To Do
# Retrain algorithm including product subset id, run on source 25 data, send to di.
# Compare some metrics to current algorithm

# Define which category you want to train for
#cat = 'CAT 11 (USE)'
cat = 'CAT 12 (EoL)'

# Load and preprocess data
d24 = TrainData()
d24.load_csv('./data/2024.csv')

# Remove excluded data and down-select columns for training
d24.df = d24.df[~d24.df[' COMMENTS '].str.contains('exclude', case = False, na = False)]
d24.df = d24.df[~d24.df['CAT 11 (USE)'].str.contains('exclude', case = False, na = False)]
d24.df = d24.df[~d24.df['CAT 12 (EoL)'].str.contains('exclude', case = False, na = False)]
keep_columns = ['SEG', 'BU', 'Business Unit', 'Sub BU', 'Sub Business Unit', 'Platform',
                'Product Category ID', 'Product Category', 'Product Line ID',
                'Product Set ID', 'Platform ID', 'Product Line', 'Product Set',
                'Material Description', 'Product Subset ID', 'Product Subset', cat]
d24.df = d24.df[keep_columns]
d24.transform()
X = d24.X
y = d24.y

# Train model
model, vectorizer = tr.train_model(X, y)

# Save model
output_dir = r'C:\Users\10354191\OneDrive - BD\Projects\SMTI\GHG\models'
model_name = 'model_24_cat12_ff.pkl'
tr.save_model(model, output_dir=output_dir, model_name=model_name)

# Save vectorizer
output_dir = r'C:\Users\10354191\OneDrive - BD\Projects\SMTI\GHG\models'
vectorizer_name = 'vec_24_cat12_ff.pkl'
tr.save_vectorizer(vectorizer, output_dir=output_dir, vectorizer_name=vectorizer_name)