import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load and split data
DATA_PATH = './data/ghg_filtered.csv'
df = pd.read_csv(DATA_PATH)
X = df['Product Subset'] + ', ' + df['Product Subset'] + ', ' + df['Product Subset'] + ', ' + df['Material Description']
y = df.iloc[:, -1]
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)
X_train, X_valtest, y_train, y_valtest = train_test_split(X, y, test_size=0.2, random_state=123)
X_val, X_test, y_val, y_test = train_test_split(X_valtest, y_valtest, test_size=0.5, random_state=123)

rfc = RandomForestClassifier(class_weight='balanced', n_estimators=100, random_state=123)
rfc.fit(X_train, y_train)

y_pred = rfc.predict(X_val)

report = classification_report(y_val, y_pred)

print(report)



