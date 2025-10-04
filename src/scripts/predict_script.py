import src.use_model as use_model
import src.Data as rt

# Predict
FILE_PATH = './data/GHG FY24 Data for MC.xlsx'
SHEET_NAME = 'FY 24 EXEMPLAR categorizations'
COLUMNS = 'A, B, F, H, L, M, Q, X'
HEADER = 2
new_X = rt.predict_data(FILE_PATH, SHEET_NAME, COLUMNS, HEADER)
model = use_model.load_model('./models/model.pkl')
vectorizer = use_model.load_vectorizer('./models/vectorizer.pkl')
preds = use_model.predict(model, vectorizer, new_X)