import pickle

# Specify the path to your .pkl file
model_path = './models/model.pkl'
vec_path = './models/vectorizer.pkl'

def load_model(model_path):
    print('Loading model')
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model

def load_vectorizer(vec_path):
    print('Loading vectorizer')
    with open(vec_path, 'rb') as file:
        vectorizer = pickle.load(file)
    return vectorizer

def predict(model, vectorizer, X):
    print('Predicting')
    # Vectorizer text
    X = vectorizer.transform(X)
    # Make predictions
    preds = model.predict(X)
    print('Prediction complete')
    return preds
