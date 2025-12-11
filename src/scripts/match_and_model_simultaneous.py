import pandas as pd
from src.data import PredictData
import src.predicting as pred

# Some paths at the top
ref_path = r"./data/2024.csv"
tar_path = r"./data/2025_source.csv"
model_path_cat11 = r"C:\Users\10354191\OneDrive - BD\Projects\SMTI\GHG\models\model_24_cat11_ff.pkl"
model_path_cat12 = r"C:\Users\10354191\OneDrive - BD\Projects\SMTI\GHG\models\model_24_cat12_ff.pkl"
vec_path_cat11 = r"C:\Users\10354191\OneDrive - BD\Projects\SMTI\GHG\models\vec_24_cat11_ff.pkl"
vec_path_cat12 = r"C:\Users\10354191\OneDrive - BD\Projects\SMTI\GHG\models\vec_24_cat12_ff.pkl"
results_path = r'./results/2025_results_ff+source+24exc.csv'

# Columns for categorization
X_columns = ['SEG', 'BU', 'Business Unit', 'Sub BU', 'Sub Business Unit', 'Platform',
                'Product Category ID', 'Product Category', 'Product Line ID',
                'Product Set ID', 'Platform ID', 'Product Line', 'Product Set',
                'Material Description', 'Product Subset ID', 'Product Subset']

# Load data
ref = pd.read_csv(ref_path)
tar = pd.read_csv(tar_path)

# Create variables for columns
cat11 = 'CAT 11 (USE)'
cat12 = 'CAT 12 (EoL)'
matid = 'Material ID'
comm = ' COMMENTS '
unit = ' FY 25 Actual Units '

# Label category excluded based on comments in the ref data
#tar.loc[tar[comm].str.contains('exclude', case = False), [cat11, cat12]] = 'excluded'

# Remove rows that contain 'excluded' in comments, cat11, and cat12 from ref data
ref[comm] = ref[comm].astype(str)
# ref = ref[~ref[comm].str.contains('exclude', case = False)]
# ref = ref[~ref[cat11].str.contains('exclude', case = False)]
# ref = ref[~ref[cat12].str.contains('exclude', case = False)]
ref[cat11] = ref[cat11].str.lower()
ref[cat12] = ref[cat12].str.lower()

# Now, branch off the subset of rows that are not excluded
#tar[comm] = tar[comm].astype(str)
#tar_inc = tar[~tar[comm].str.contains('exclude', case = False)]
tar_inc = tar

# Convert Material ID column to string
tar_inc[matid] = tar_inc[matid].astype(str)
ref[matid] = ref[matid].astype(str)
ref.drop_duplicates(subset=matid, inplace = True)

# Match/merge on Material ID
match = tar_inc.merge(ref[[matid, cat11, cat12]], on = matid, how = 'left')
tar_inc[[cat11, cat12]] = match[[cat11, cat12]].values
tar_inc.loc[~tar_inc[cat11].isna(), comm] = 'Matched by Material ID'

# Now, branch off the remaining, unmatched rows for machine learning
tar_unmatched = tar_inc[tar_inc[cat11].isna()]

# Classify unmatched rows based on trained model
tar_data = PredictData()
tar_data.df = tar_unmatched[X_columns]
tar_data.transform()

# Load models for inference
model_cat11 = pred.load_model(model_path_cat11)
model_cat12 = pred.load_model(model_path_cat12)

# Load vectorizer
vec_cat11 = pred.load_vectorizer(vec_path_cat11)
vec_cat12 = pred.load_vectorizer(vec_path_cat12)

# Run prediction
preds_cat11 = pred.predict(model_cat11, vec_cat11, tar_data.X)
preds_cat12 = pred.predict(model_cat12, vec_cat12, tar_data.X)
tar_unmatched[cat11] = preds_cat11.preds.values
tar_unmatched[cat12] = preds_cat12.preds.values
tar_unmatched.loc[~tar_unmatched[cat11].isna(), comm] = 'Classified by ML model'

# Relabel primary df
tar.iloc[tar_inc.index] = tar_inc
tar.iloc[tar_unmatched.index] = tar_unmatched

# Check duplicate Material IDs in unmatched data
matid_dup = tar.loc[tar[matid].duplicated(), matid].unique()

# Export results
tar.to_csv(results_path)





# units = tar[unit]
# units = units.str.strip()
# units[units.str.contains('-')] = '0'
# units = units.str.strip('(')
# units = units.str.strip(')')
# units = units.str.replace(',', '')