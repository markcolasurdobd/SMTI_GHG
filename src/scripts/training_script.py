import pandas as pd
import src.training as tr
from src.datasets import make_2024

# Read in data
d24 = make_2024()
exclude = d24.y[d24.y == 'exclude'].index
d24.X = d24.X.drop(exclude)
d24.y = d24.y.drop(exclude)
X = d24.X
y = d24.y

# Train model
model, vectorizer = tr.train_model(X, y)

# Save model
output_dir = r'C:\Users\10354191\OneDrive - BD\Projects\SMTI\GHG\models'
model_name = 'model_24.pkl'
tr.save_model(model, output_dir=output_dir, model_name=model_name)

# Save vectorizer
output_dir = r'C:\Users\10354191\OneDrive - BD\Projects\SMTI\GHG\models'
vectorizer_name = 'vec_24.pkl'
tr.save_vectorizer(vectorizer, output_dir=output_dir, vectorizer_name=vectorizer_name)