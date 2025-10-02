from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

def train_model(X, y, model_name=None, vec_name=None):
    print("Training vectorizer")
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(X)

    print("Training model")
    model = RandomForestClassifier(class_weight='balanced', n_estimators=100, random_state=123)
    model.fit(X, y)
    print("Training complete")

    # Save the model and vectorizer
    output_dir = './models/'
    if model_name is None:
        model_name = 'model.pkl'
        path = os.path.join(output_dir, model_name)
        with open(path, 'wb') as file:
            pickle.dump(model, file)
    else:
        path = os.path.join(output_dir, model_name)
        with open(path, 'wb') as file:
            pickle.dump(model, file)
    print(f"Model saved to {path}")

    if vec_name is None:
        vec_name = 'vectorizer.pkl'
        path = os.path.join(output_dir, vec_name)
        with open(path, 'wb') as file:
            pickle.dump(vectorizer, file)
    else:
        path = os.path.join(output_dir, vec_name)
        with open(path, 'wb') as file:
            pickle.dump(vectorizer, file)
    print(f"Vectorizer saved to {path}")

    return model, vectorizer