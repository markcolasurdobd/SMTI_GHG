import pandas as pd
import src.training as train
import src.predicting as pred
from sklearn.metrics import classification_report

# Load data
df_train =  pd.read_csv('./data/train.csv')
df_val = pd.read_csv('./data/validation.csv')

# Split df into X and y
X_train, y_train = df_train.iloc[:, 0], df_train.iloc[:, -1]
X_val, y_val = df_val.iloc[:, 0], df_val.iloc[:, -1]

# Train the model
model, vectorizer = train.train_model(X_train, y_train)

# Predict on X_val
y_preds = pred.predict(model, vectorizer, X_val)

# Run classification report
cr = classification_report(y_val, y_preds, output_dict=False)
print(cr)

# Save classification report
cr = classification_report(y_val, y_preds, output_dict=True)
df_cr = pd.DataFrame(cr).transpose()
df_cr.to_csv('./results/validation_results_21_24_unbalanced.csv')