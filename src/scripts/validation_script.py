from src.data import TrainData
import pandas as pd
import src.training as train
import src.predicting as pred
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load and preprocess data
df =  pd.read_csv('./data/2024_w_cat12.csv')
exclude_rows = df[' COMMENTS '].str.contains('exclude', case = False, na = False)
df_included = df[~exclude_rows]
d24 = TrainData()
X_cols = ['SEG', 'BU', 'Platform', 'Product Category', 'Product Line', 'Product Set']
y_col = ['CAT 11 (USE)']
d24.df = df_included[X_cols + y_col]
d24.transform()

# Split df into X and y
X_train, X_test, y_train, y_test = train_test_split(d24.X, d24.y, test_size=0.2, random_state=42)

# Train the model
model, vectorizer = train.train_model(X_train, y_train)

# Predict on X_val
y_preds = pred.predict(model, vectorizer, X_test)

# Run classification report
cr = classification_report(y_test, y_preds.preds, output_dict=True)
#print(cr)

# Save results
df_report = pd.DataFrame(cr).transpose()
df_report.to_csv('./results/2024_cat11_report.csv')