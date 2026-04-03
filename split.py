

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

os.makedirs('outputs', exist_ok=True)


print("Loading cleaned data...")
df = pd.read_csv('outputs/clean_data.csv')
print(f"Shape: {df.shape}")


X = df.drop(columns=['churn'])
y = df['churn']

print(f"\nFeatures : {X.shape[1]}")
print(f"Samples  : {X.shape[0]:,}")
print(f"Churn rate: {y.mean()*100:.1f}%")



print(f"\n{'='*55}")
print("SPLIT 1 — RANDOM SHUFFLE (THE WRONG WAY)")
print(f"{'='*55}")

X_train_rand, X_test_rand, y_train_rand, y_test_rand = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    shuffle=True        
)

print(f"Train size : {len(X_train_rand):,}")
print(f"Test size  : {len(X_test_rand):,}")
print(f"Train churn rate: {y_train_rand.mean()*100:.1f}%")
print(f"Test churn rate : {y_test_rand.mean()*100:.1f}%")
print(f"\nNotice: churn rates are almost identical in train and test.")
print(f"That's because shuffle mixed future+past together evenly.")
print(f"This feels 'balanced' but it's leaking future information.")



print(f"\n{'='*55}")
print("SPLIT 2 — TIME-BASED (THE RIGHT WAY)")
print(f"{'='*55}")

split_idx = int(len(df) * 0.80)   

X_train_time = X.iloc[:split_idx]   
X_test_time  = X.iloc[split_idx:]   
y_train_time = y.iloc[:split_idx]
y_test_time  = y.iloc[split_idx:]

print(f"Split at row   : {split_idx:,}")
print(f"Train size     : {len(X_train_time):,}")
print(f"Test size      : {len(X_test_time):,}")
print(f"Train churn rate: {y_train_time.mean()*100:.1f}%")
print(f"Test churn rate : {y_test_time.mean()*100:.1f}%")
print(f"\nNotice: churn rates may differ between train and test.")
print(f"That's REAL. Different time periods have different patterns.")
print(f"This is what honest evaluation looks like.")


print(f"\n{'='*55}")
print("SPLIT SUMMARY")
print(f"{'='*55}")
print(f"Random split  → train/test are shuffled mixed-time slices")
print(f"Time split    → train=past rows, test=future rows")
print(f"\nIn train.py we will train 7 models on BOTH splits.")
print(f"The AUC difference between them = temporal leakage delta.")


X_train_rand.to_csv('outputs/X_train_rand.csv', index=False)
X_test_rand.to_csv('outputs/X_test_rand.csv',   index=False)
y_train_rand.to_csv('outputs/y_train_rand.csv', index=False)
y_test_rand.to_csv('outputs/y_test_rand.csv',   index=False)

X_train_time.to_csv('outputs/X_train_time.csv', index=False)
X_test_time.to_csv('outputs/X_test_time.csv',   index=False)
y_train_time.to_csv('outputs/y_train_time.csv', index=False)
y_test_time.to_csv('outputs/y_test_time.csv',   index=False)

print(f"\n All 8 split files saved to outputs/")
print(f"Next: run train.py to train 7 models on both splits")