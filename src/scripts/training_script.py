import pandas as pd
from src.data import TrainData
import src.training as tr

# Load and preprocess data
d24 = TrainData()
d24.load_csv('./data/2024_w_cat12.csv')
keep_columns = ['SEG', 'BU', 'Platform', 'Product Category', 'Product Line', 'Product Set', 'CAT 12 (EoL)']
d24.df = d24.df[keep_columns]
d24.transform()
d24.remove_value('blank', d24.X)
d24.replace_substring('exclude')
exclude = d24.y[d24.y == 'exclude'].index
d24.X = d24.X.drop(exclude)
d24.y = d24.y.drop(exclude)
X = d24.X
y = d24.y

# Train model
model, vectorizer = tr.train_model(X, y)

# Save model
output_dir = r'C:\Users\10354191\OneDrive - BD\Projects\SMTI\GHG\models'
model_name = 'model_24_cat12.pkl'
tr.save_model(model, output_dir=output_dir, model_name=model_name)

# Save vectorizer
output_dir = r'C:\Users\10354191\OneDrive - BD\Projects\SMTI\GHG\models'
vectorizer_name = 'vec_24_cat12.pkl'
tr.save_vectorizer(vectorizer, output_dir=output_dir, vectorizer_name=vectorizer_name)