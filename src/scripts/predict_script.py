import src.predicting as pred

# Load data

# Load model
model_path = ''
model = pred.load_model(model_path)

# Load vectorizer
vec_path = ''
vectorizer = pred.load_vectorizer(vec_path)

# Make predictions
preds = pred.predict(model, vectorizer, X)