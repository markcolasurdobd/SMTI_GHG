import src.use_model as use_model
import src.read_and_transform as rt

# Predict
new_X = rt.predict_data(FILE_PATH, SHEET_NAME, COLUMNS, HEADER)
model = use_model.load_model('./models/model.pkl')
vectorizer = use_model.load_vectorizer('./models/vectorizer.pkl')
preds = use_model.predict(model, vectorizer, new_X)