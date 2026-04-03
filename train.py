# train.py — KDD Churn Oracle



import pandas as pd
import numpy as np
import pickle
import os
import time

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

os.makedirs('outputs/models', exist_ok=True)


print("Loading splits...")

X_train_rand = pd.read_csv('outputs/X_train_rand.csv')
X_test_rand  = pd.read_csv('outputs/X_test_rand.csv')
y_train_rand = pd.read_csv('outputs/y_train_rand.csv').squeeze()
y_test_rand  = pd.read_csv('outputs/y_test_rand.csv').squeeze()

X_train_time = pd.read_csv('outputs/X_train_time.csv')
X_test_time  = pd.read_csv('outputs/X_test_time.csv')
y_train_time = pd.read_csv('outputs/y_train_time.csv').squeeze()
y_test_time  = pd.read_csv('outputs/y_test_time.csv').squeeze()

print(f"Random split — train: {len(X_train_rand):,}  test: {len(X_test_rand):,}")
print(f"Time split   — train: {len(X_train_time):,}  test: {len(X_test_time):,}")



models = [
    (
        "1_LogisticRegression",
        LogisticRegression(max_iter=1000, random_state=42)
    ),
    (
        "2_DecisionTree_depth3",
        DecisionTreeClassifier(max_depth=3, random_state=42)
    ),
    (
        "3_DecisionTree_depth10",
        DecisionTreeClassifier(max_depth=10, random_state=42)
    ),
    (
        "4_DecisionTree_unlimited",
        DecisionTreeClassifier(max_depth=None, random_state=42)
    ),
    (
        "5_RandomForest",
        RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    ),
    (
        "6_GradientBoosting",
        GradientBoostingClassifier(n_estimators=100, random_state=42)
    ),
    (
        "7_XGBoost",
        XGBClassifier(
            n_estimators=100,
            random_state=42,
            eval_metric='logloss',
            verbosity=0
        )
    ),
]



print(f"\n{'='*55}")
print("TRAINING ALL 7 MODELS ON BOTH SPLITS")
print(f"{'='*55}")
print(f"{'Model':<30} {'Split':<10} {'Time':>8}")
print("-"*55)

for name, model in models:

 
    import copy
    model_rand = copy.deepcopy(model)   
    t0 = time.time()
    model_rand.fit(X_train_rand, y_train_rand)
    t1 = time.time()
    print(f"{name:<30} {'random':<10} {t1-t0:>6.1f}s")

    
    with open(f'outputs/models/{name}_rand.pkl', 'wb') as f:
        pickle.dump(model_rand, f)

    
    model_time = copy.deepcopy(model)   
    t0 = time.time()
    model_time.fit(X_train_time, y_train_time)
    t1 = time.time()
    print(f"{name:<30} {'time':<10} {t1-t0:>6.1f}s")

    
    with open(f'outputs/models/{name}_time.pkl', 'wb') as f:
        pickle.dump(model_time, f)

print(f"\n{'='*55}")
print(f" Training complete.")
print(f"   14 models saved to outputs/models/")
print(f"   (7 models × 2 splits = 14 total)")
print(f"\nNext: run evaluate.py to measure AUC and leakage delta")