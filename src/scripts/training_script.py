import pandas as pd
from sklearn.model_selection import train_test_split

from src.train_model import train_model

# Read in data
df = pd.read_csv('./data/training_data_21+24.csv')
X = df['features']
y = df['target']

# Train model
train_model(X, y, model_name='model_21+24.pkl', vec_name='vectorizer_21+24')
