# eda.py — KDD Churn Oracle
# Full exploratory data analysis
 
import pandas as pd
import numpy as np
import os

os.makedirs('outputs', exist_ok=True)


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

print(f"Data rows   : {len(df)}")
print(f"Label rows  : {len(labels)}")

df['churn'] = labels['churn'].values


print("\n" + "="*55)
print("SECTION 1 — SHAPE AND FIRST LOOK")
print("="*55)
print(f"Rows    : {df.shape[0]:,}")
print(f"Columns : {df.shape[1]:,}  (230 features + 1 target)")
print(f"\nFirst 3 rows, first 6 columns:")
print(df.iloc[:3, :6])


print("\n" + "="*55)
print("SECTION 2 — TARGET DISTRIBUTION")
print("="*55)
total   = len(df)
churned = (df['churn'] == 1).sum()
stayed  = (df['churn'] == -1).sum()
print(f"Stayed  (-1) : {stayed:,}  ({stayed/total*100:.1f}%)")
print(f"Churned (+1) : {churned:,}  ({churned/total*100:.1f}%)")
print(f"Imbalance ratio: {stayed//churned}:1  (stayed:churned)")

print("\n" + "="*55)
print("SECTION 2 — MISSING VALUES")
print("="*55)
features    = df.drop(columns=['churn'])
missing_pct = (features.isnull().sum() / len(df) * 100).round(1)
print(f"Columns with 0% missing    : {(missing_pct == 0).sum()}")
print(f"Columns with >50% missing  : {(missing_pct > 50).sum()}")
print(f"Columns with >80% missing  : {(missing_pct > 80).sum()}")
print(f"Columns with 100% missing  : {(missing_pct == 100).sum()}")
print(f"Avg missingness            : {missing_pct.mean():.1f}%")
print(f"\nTop 10 most empty columns:")
for col, pct in missing_pct.sort_values(ascending=False).head(10).items():
    print(f"  {col:10s}  {pct:.1f}%")


print("\n" + "="*55)
print("SECTION 3 — COLUMN TYPES")
print("="*55)
num_cols   = [f'Var{i}' for i in range(1, 191)]
cat_cols   = [f'Var{i}' for i in range(191, 231)]
ghost_cols = missing_pct[missing_pct > 80].index.tolist()
print(f"Numerical columns   : {len(num_cols)}  (Var1–Var190)")
print(f"Categorical columns : {len(cat_cols)}  (Var191–Var230)")
print(f"Ghost cols (>80%)   : {len(ghost_cols)}")


report = [
    "KDD CHURN ORACLE — EDA REPORT",
    "="*55,
    f"Rows           : {df.shape[0]:,}",
    f"Columns        : {df.shape[1]:,}",
    f"Churned (+1)   : {churned:,} ({churned/total*100:.1f}%)",
    f"Stayed  (-1)   : {stayed:,} ({stayed/total*100:.1f}%)",
    f"Avg missing    : {missing_pct.mean():.1f}%",
    f"Ghost cols >80%: {len(ghost_cols)}",
    "", "Ghost columns:"
] + [f"  {c}" for c in ghost_cols]

with open('outputs/eda_report.txt', 'w') as f:
    f.write('\n'.join(report))

print("\n EDA complete. Report saved to outputs/eda_report.txt")
