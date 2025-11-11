import pandas as pd
from src.data import ValidationData
import src.predicting as pred
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# Load results
df = pd.read_csv('./data/2025_results.csv')

# Pull results that were matched by Material ID and CAT 11 is not excluded
mat_id = df[df[' COMMENTS '] == 'Matched by 2024 Material ID']
mat_id = mat_id[mat_id['CAT 11'] != 'EXCLUDE']

# Keep only necessary columns
cols = ['SEG', 'BU', 'Platform', 'Product Category', 'Product Line', 'Product Set', 'CAT 11']
mat_id = mat_id[cols]

# Convert to ValidationData class
mat_data = ValidationData()
mat_data.df = mat_id
mat_data.transform()

# Load models
model_path = r"C:\Users\10354191\OneDrive - BD\Projects\SMTI\GHG\models\model_24.pkl"
vec_path = r"C:\Users\10354191\OneDrive - BD\Projects\SMTI\GHG\models\vec_24.pkl"
model = pred.load_model(model_path)
vec = pred.load_vectorizer(vec_path)

# Run models
preds = pred.predict(model, vec, mat_data.X)
cr = classification_report(mat_data.y, preds.preds)
print(cr)

# Append predictions to df
preds.index = mat_id.index
mat_data.y.index = preds.index
mat_id['preds'] = preds.preds
mat_id['probs'] = preds.probs

# Find the mismatched lines
wrong = mat_id[mat_data.y != preds.preds]
right = mat_id[mat_data.y == preds.preds]
wrong.to_csv('./results/wrong.csv')

# Plot wrong histograms
plt.hist(wrong.probs)
plt.title("Prediction confidence for incorrect predictions")
plt.ylabel("Count")
plt.xlabel('Prediction Confidence')
plt.show()

# Plot right histograms
plt.hist(right.probs)
plt.title("Prediction confidence for correct predictions")
plt.ylabel("Count")
plt.xlabel('Prediction Confidence')
plt.show()