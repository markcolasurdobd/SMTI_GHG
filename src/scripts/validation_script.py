import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report

# Load data
df_train =  pd.read_csv('./data/train.csv')
df_val = pd.read_csv('./data/validation.csv')
X_train, y_train = df_train.iloc[:, 0], df_train.iloc[:, -1]
X_val, y_val = df_val.iloc[:, 0], df_val.iloc[:, -1]

# Instantiate vectorizer and model
vectorizer = TfidfVectorizer()
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Run vectorizer and model
X_train = vectorizer.fit_transform(X_train)
model.fit(X_train, y_train)
X_val = vectorizer.transform(X_val)
y_preds = model.predict(X_val)

# Run classification report
cr = classification_report(y_val, y_preds, output_dict=True)
print(cr)
df_cr = pd.DataFrame(cr).transpose()
df_cr.to_csv('./results/train_and_validation_results_21_24.csv')









