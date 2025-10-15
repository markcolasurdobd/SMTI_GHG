import pandas as pd
import src.training as tr

# Read in data
df = pd.read_csv('./data/master.csv')
X = df.iloc[:, 0]
y = df.iloc[:, -1]

# Train model
model, vectorizer = tr.train_model(X, y)

# Save model
output_dir = r'C:\Users\10354191\OneDrive - BD\Projects\SMTI\GHG\models'
model_name = 'model_21_24.pkl'
tr.save_model(model, output_dir=output_dir, model_name=model_name)

# Save vectorizer
output_dir = r'C:\Users\10354191\OneDrive - BD\Projects\SMTI\GHG\models'
vectorizer_name = 'vec_21_24.pkl'
tr.save_vectorizer(vectorizer, output_dir=output_dir, vectorizer_name=vectorizer_name)