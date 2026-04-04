# clean.py — KDD Churn Oracle
 

import pandas as pd
import numpy as np
import os

os.makedirs('outputs', exist_ok=True)


print("Loading raw data...")
df = pd.read_csv(
    'data/orange_small_train.data',
    sep='\t',
    header=0,
    low_memory=False
)
df.columns = [f'Var{i+1}' for i in range(df.shape[1])]
df = df.dropna(how='all').reset_index(drop=True)

labels = pd.read_csv(
    'data/orange_small_train_churn.labels',
    header=None,
    names=['churn']
)
labels = labels.dropna().reset_index(drop=True)
df['churn'] = labels['churn'].values

print(f"Raw shape: {df.shape}")



features = df.drop(columns=['churn'])
missing_pct = features.isnull().sum() / len(df)
ghost_cols = missing_pct[missing_pct > 0.80].index.tolist()

df = df.drop(columns=ghost_cols)
print(f"\nStep 1 — Dropped {len(ghost_cols)} ghost columns")
print(f"Remaining columns: {df.shape[1]} (including target)")



num_cols = [c for c in df.columns
            if c.startswith('Var')
            and int(c[3:]) <= 190]

cat_cols = [c for c in df.columns
            if c.startswith('Var')
            and int(c[3:]) > 190]

print(f"\nStep 2 — Remaining after ghost removal:")
print(f"  Numerical  : {len(num_cols)}")
print(f"  Categorical: {len(cat_cols)}")



print(f"\nStep 3 — Imputing numerical columns with median...")
for col in num_cols:
    median_val = df[col].median()
    df[col] = df[col].fillna(median_val)

num_missing_after = df[num_cols].isnull().sum().sum()
print(f"  Missing values in numerical cols after imputation: {num_missing_after}")



print(f"\nStep 4 — Imputing categorical columns with mode...")
for col in cat_cols:
    if df[col].isnull().sum() > 0:
        mode_val = df[col].mode()
        if len(mode_val) > 0:
            df[col] = df[col].fillna(mode_val[0])

cat_missing_after = df[cat_cols].isnull().sum().sum()
print(f"  Missing values in categorical cols after imputation: {cat_missing_after}")



print(f"\nStep 5 — Encoding categorical columns...")
for col in cat_cols:
    
    df[col] = df[col].astype(str)
    
    unique_vals = df[col].unique()
    mapping = {val: idx for idx, val in enumerate(unique_vals)}
    df[col] = df[col].map(mapping)

print(f"  Categorical columns encoded: {len(cat_cols)}")



df['churn'] = df['churn'].map({-1: 0, 1: 1})
print(f"\nStep 6 — Target converted: -1→0, +1→1")
print(f"  Churn distribution after conversion:")
print(f"  {df['churn'].value_counts().to_dict()}")


total_missing = df.drop(columns=['churn']).isnull().sum().sum()
print(f"\n{'='*55}")
print(f"CLEAN SUMMARY")
print(f"{'='*55}")
print(f"Final shape        : {df.shape}")
print(f"Total missing left : {total_missing}")
print(f"Columns kept       : {df.shape[1]-1} features + 1 target")


df.to_csv('outputs/clean_data.csv', index=False)
print(f"\n Cleaned data saved to outputs/clean_data.csv")
