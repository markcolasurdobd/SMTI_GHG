from difflib import SequenceMatcher
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data
df23 = pd.read_csv('./data/ghg_filtered_2023.csv')
df24 = pd.read_csv('./data/ghg_filtered_2024.csv')

KEEP_24 = ['SEG', 'BU', 'Platform', 'Product Category', 'Product Line', 'Product Set', 'CAT 11 (USE)']

df24 = df24[KEEP_24]

df24['string'] = df24['SEG'].str.cat(df24[df24.columns[1:-1]].astype(str), sep=' ')

seq1 = df24.string[1]
seq2 = df24.string[2]

ratios = [SequenceMatcher(None, seq1, seq).ratio() for seq in df24.string]

plt.hist(ratios)
plt.show()