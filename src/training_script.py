import src.read_and_transform as rt
from src import use_model
from src.train_model import train_model
import src.use_model

# Set input variables
FILE_PATH = './data/GHG FY24 Data for MC.xlsx'
SHEET_NAME = 'FY 24 EXEMPLAR categorizations'
COLUMNS = 'A, B, F, H, L, M, Q, X'
HEADER = 2

# Read and transform data
X, y = rt.training_data(FILE_PATH, SHEET_NAME, COLUMNS, HEADER)

# Train model
train_model(X, y)
