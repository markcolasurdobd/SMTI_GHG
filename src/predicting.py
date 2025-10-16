import numpy as np
import pandas as pd
import pickle

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
    probs_arr = model.predict_proba(X)
    class_probs = []
    for i, label in enumerate(preds):
        class_idx = np.where(model.classes_ == label)
        class_idx = int(class_idx[0][0])
        class_probs.append(probs_arr[i][class_idx])
    output_df = pd.DataFrame([preds, class_probs])
    output_df = output_df.transpose()
    output_df.columns = ['preds', 'probs']
    print('Prediction complete')
    return output_df



