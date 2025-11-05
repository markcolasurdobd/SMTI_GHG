import pandas as pd

def filter_df(df, col, str):
    return df[~df[col].str.contains(str, case = False)]

# Load csv
path = r"./data/2025.csv"
df = pd.read_csv(path)

# Filter Product Category
to_filter = ['software', 'sw', 'do not use']
for word in to_filter:
    df = filter_df(df, "Product Category", word)


# Filter Product Line
to_filter = ['services', 'fees', 'incentive']
for word in to_filter:
    df = filter_df(df, "Product Line", word)

# Filter Product Set
to_filter = ['do not use', 'tie out', 'service', 'agreement', 'contract', 'shipping', 'handling', 'freight', 'royalty',
             'royalties', 'fee ', 'fees', 'time and materials', 'maintenance', 'software', 'training', 'trade', 'rental',
             'rebate', 'unknown', 'operating', 'labor']
for word in to_filter:
    df = filter_df(df, 'Product Set', word)

# Filter Material ID for MMS and PS
mms_drop = df[(df['BU'] == 'MMS') & (df['Material ID'].str.contains('PC'))].index
ps_drop = df[(df['BU'] == 'PS') & (df['Material ID'].str.contains('PC'))].index
df = df.drop(mms_drop)
df = df.drop(ps_drop)

# Filter Material Description
to_filter = ['dummy', 'exclude']
for word in to_filter:
    df = filter_df(df, 'Material Description', word)

# Filter Comments
comments_not_na = df[~df[' COMMENTS '].isna()]
comments_drop = comments_not_na[comments_not_na[' COMMENTS '].str.contains('exclude', case = False)].index
df = df.drop(comments_drop)

# Material ID column to string
df['Material ID'] = df['Material ID'].astype(str)

# Send to csv
out_path = './data/2025_filtered.csv'
df.to_csv(out_path)

# Read in 2024 data
df_24 = pd.read_csv('./data/2024.csv')