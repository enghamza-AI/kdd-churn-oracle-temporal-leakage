# evaluate.py — KDD Churn Oracle
 

import pandas as pd
import numpy as np
import pickle
import os

from sklearn.metrics import roc_auc_score

os.makedirs('outputs', exist_ok=True)


print("Loading test sets...")

X_test_rand  = pd.read_csv('outputs/X_test_rand.csv')
y_test_rand  = pd.read_csv('outputs/y_test_rand.csv').squeeze()

X_test_time  = pd.read_csv('outputs/X_test_time.csv')
y_test_time  = pd.read_csv('outputs/y_test_time.csv').squeeze()


model_names = [
    "1_LogisticRegression",
    "2_DecisionTree_depth3",
    "3_DecisionTree_depth10",
    "4_DecisionTree_unlimited",
    "5_RandomForest",
    "6_GradientBoosting",
    "7_XGBoost",
]


display_names = [
    "LogReg",
    "Tree d=3",
    "Tree d=10",
    "Tree ∞",
    "RandomForest",
    "GradBoost",
    "XGBoost",
]



print(f"\n{'='*65}")
print(f"{'Model':<25} {'AUC Random':>12} {'AUC Time':>10} {'Delta':>8}")
print(f"{'':.<25} {'(wrong)':>12} {'(honest)':>10} {'(leak)':>8}")
print("="*65)

results = []

for name, display in zip(model_names, display_names):

    
    with open(f'outputs/models/{name}_rand.pkl', 'rb') as f:
        model_rand = pickle.load(f)

    
    with open(f'outputs/models/{name}_time.pkl', 'rb') as f:
        model_time = pickle.load(f)

  
    prob_rand = model_rand.predict_proba(X_test_rand)[:, 1]
    prob_time = model_time.predict_proba(X_test_time)[:, 1]

   
    auc_rand = roc_auc_score(y_test_rand, prob_rand)
    auc_time = roc_auc_score(y_test_time, prob_time)

    
    delta = auc_rand - auc_time

    print(f"{display:<25} {auc_rand:>12.4f} {auc_time:>10.4f} {delta:>8.4f}")

    results.append({
        'model': display,
        'auc_random': round(auc_rand, 4),
        'auc_time':   round(auc_time, 4),
        'delta':      round(delta, 4),
        'complexity': model_names.index(name) + 1
    })

print("="*65)


results_df = pd.DataFrame(results)

max_delta_row = results_df.loc[results_df['delta'].idxmax()]
min_delta_row = results_df.loc[results_df['delta'].idxmin()]

print(f"\nBiggest leaker : {max_delta_row['model']} "
      f"(delta = {max_delta_row['delta']:.4f})")
print(f"Smallest leaker: {min_delta_row['model']} "
      f"(delta = {min_delta_row['delta']:.4f})")
print(f"\nAverage AUC inflation from random split: "
      f"{results_df['delta'].mean():.4f}")


results_df.to_csv('outputs/leakage_results.csv', index=False)
print(f"\nResults saved to outputs/leakage_results.csv")
print(f"\nNext: run visualize.py to draw the complexity cliff chart")
print(f"Then: run app.py for the Streamlit dashboard")


print(f"\n{'='*65}")
print("KEY INSIGHT")
print(f"{'='*65}")
print(f"The delta column tells you the exact AUC points that were")
print(f"STOLEN by the random shuffle.")

