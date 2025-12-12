from src.data import TrainData
import pandas as pd
import src.training as train
import src.predicting as pred
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Paths
ref_path = r'./data/2024.csv'
results_path = r'./results/2024_cat12_ff_report.csv'

# Columns
comm = ' COMMENTS '
cat11 = 'CAT 11 (USE)'
cat12 = 'CAT 12 (EoL)'
y_col = [cat12]
X_cols = ['SEG', 'BU', 'Business Unit', 'Sub BU', 'Sub Business Unit', 'Platform',
                'Product Category ID', 'Product Category', 'Product Line ID',
                'Product Set ID', 'Platform ID', 'Product Line', 'Product Set',
                'Material Description', 'Product Subset ID', 'Product Subset']

# Load data
df =  pd.read_csv(ref_path)

# Preprocess data
df = df[~df[comm].str.contains('exclude', case = False, na = False)]
df = df[~df[cat11].str.contains('exclude', case = False, na = False)]
df = df[~df[cat12].str.contains('exclude', case = False, na = False)]

#  Prepare data
d24 = TrainData()
d24.df = df[X_cols + y_col]
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
df_report.to_csv(results_path)