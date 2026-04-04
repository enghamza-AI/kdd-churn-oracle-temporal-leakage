📡 KDD Churn Oracle — Temporal Leakage Trap
Live Demo: huggingface.co/spaces/enghamza-AI/Tempex

Measuring the exact cost of trusting a shuffled split — across 7 model complexities.

🔍 What This Project Does
Most ML engineers shuffle their data before splitting into train/test. This feels harmless. It is not.
When your data has a time dimension — customers recorded over months — a random shuffle lets future data leak into your training set. The model "sees the future" during training. AUC looks great. Real-world performance collapses.
This project measures that exact mistake across 7 model complexities, from logistic regression to XGBoost, and visualizes the leakage delta — the precise number of AUC points stolen by an improper split.

🧠 The Core Concept — Temporal Leakage
Random Split (WRONG):   [Jan|Apr|Dec|Feb|Nov|...]  →  shuffle  →  80/20 split
                         Future rows leak into training. Model cheats.

Time Split (RIGHT):     [Jan|Feb|Mar|...|Oct|Nov|Dec]  →  first 80% | last 20%
                         Model trained on past. Tested on future. Honest.

Leakage Delta  =  AUC(random split)  −  AUC(time split)
The more complex the model, the more aggressively it exploits leaked data. An unlimited decision tree memorizes every row — including future ones. Logistic regression draws one line — it cannot memorize. That's why the delta grows with complexity. That's the complexity cliff.

📊 Dataset
PropertyValueSourceKDD Cup 2009 — Orange TelecomRows50,000 customersFeatures230 (190 numerical, 40 categorical)TargetChurn: +1 churned, -1 stayedChurn rate~7.3% (12:1 class imbalance)Avg missingness69.8% across all columns

🏗️ Project Structure
kdd-churn-oracle-temporal-leakage/
│
├── data/
│   ├── orange_small_train.data          ← 50k rows, 230 features
│   └── orange_small_train_churn.labels  ← target labels (+1/-1)
│
├── outputs/
│   ├── clean_data.csv                   ← cleaned dataset
│   ├── complexity_cliff.png             ← main chart
│   ├── leakage_delta_bars.png           ← delta bar chart
│   └── leakage_results.csv             ← AUC scores all models
│
├── eda.py          ← exploratory data analysis
├── clean.py        ← imputation, encoding, ghost column removal
├── split.py        ← random split vs time-based split
├── train.py        ← train 7 models on both splits
├── evaluate.py     ← AUC measurement + leakage delta
├── visualize.py    ← complexity cliff + delta bar charts
├── app.py          ← Streamlit dashboard
└── requirements.txt

🧪 The 7 Models (Complexity Ladder)
#ModelWhy It's Here1Logistic RegressionSimplest — draws one straight line2Decision Tree depth=3Small flowchart, 3 decisions3Decision Tree depth=10Medium complexity4Decision Tree depth=NoneUnlimited — memorizes everything5Random Forest (100 trees)Ensemble, diverse trees6Gradient BoostingSequential error correction7XGBoostState of the art boosting
Each model is trained twice — once on the random split, once on the time split. The AUC difference is the leakage delta.

🔑 Key Findings
The complexity cliff chart shows what theory predicts and data confirms:

Simple models (logistic regression) have small leakage deltas — they cannot memorize future rows aggressively enough to exploit the leak
Complex models (unlimited tree, XGBoost) have large leakage deltas — they memorize every pattern including leaked future data
The average AUC inflation across all 7 models demonstrates how much a random shuffle silently lies to you


🚀 Run It Yourself
bash# Clone
git clone https://github.com/enghamza-AI/kdd-churn-oracle-temporal-leakage
cd kdd-churn-oracle-temporal-leakage

# Install
pip install -r requirements.txt

# Download dataset from:
# https://www.kdd.org/kdd-cup/view/kdd-cup-2009/Data
# Place files in data/ folder

# Run pipeline in order
python eda.py
python clean.py
python split.py
python train.py
python evaluate.py
python visualize.py

# Launch app
streamlit run app.py

💡 What I Learned Building This

Temporal leakage is one of the most common and costly mistakes in production ML — and one of the least discussed in tutorials
Data cleaning at scale — 69.8% average missingness requires median imputation, not row deletion
Class imbalance (12:1 ratio) makes accuracy useless as a metric — AUC is the right tool
Model complexity and overfitting — unlimited depth trees memorize noise AND leaked future data simultaneously
Why visualization is the real deliverable — a chart communicates what a CSV of numbers cannot


🗂️ What's Next
This is Stage 1, Week 3 of my AI Systems Engineering roadmap. The full roadmap targets roles at Anthropic, xAI, OpenAI, Perplexity, and YC-backed startups by early 2028.
Previous projects:

Vexis — Clinical AI corruption diagnostic tool (MIMIC-III, 758k rows)
NYC Zonal Prediction Failure Atlas


👤 Author
Hamza — BSAI student, self-studying AI Systems Engineering
GitHub: @enghamza-AI
