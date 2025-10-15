from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

def train_model(X, y):
    print("Training vectorizer")
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(X)

    print("Training model")
    model = RandomForestClassifier(class_weight='balanced', n_estimators=100, random_state=123)
    model.fit(X, y)
    print("Training complete")
    return model, vectorizer

def save_model(model, output_dir=os.getcwd(), model_name='model.pkl'):
    print(f"Saving model as {model_name} to {output_dir}")
    path = os.path.join(output_dir, model_name)
    with open(path, 'wb') as file:
        pickle.dump(model, file)

def save_vectorizer(vectorizer, output_dir=os.getcwd(), vectorizer_name='vectorizer.pkl'):
    print(f"Saving vectorizer as {vectorizer_name} to {output_dir}")
    path = os.path.join(output_dir, vectorizer_name)
    with open(path, 'wb') as file:
        pickle.dump(vectorizer, file)