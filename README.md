# 📡 KDD Churn Oracle — Temporal Leakage Trap

**Measuring the exact cost of trusting a shuffled split — across 7 model complexities.**

### 🎯 Live Demo
[![Live Demo](https://img.shields.io/badge/🚀_Live_Demo-FF6B6B?style=for-the-badge&logo=streamlit&logoColor=white)](https://huggingface.co/spaces/enghamza-AI/Tempex)

---

### 🔍 The Problem Most People Ignore

Most ML engineers **shuffle** their data before splitting into train/test.  
It feels harmless. **It is not.**

When your data has a **time dimension** (customers recorded over months), a random shuffle lets **future data leak into the training set**.  
The model "sees the future" during training.  
**AUC looks amazing** on the test set.  
**Real-world performance collapses.**

This project quantifies that exact mistake.

---

### 🧠 Core Concept: Temporal Leakage

**Random Split (WRONG)**  
`[Jan|Apr|Dec|Feb|Nov|...]` → shuffle → 80/20 split  
→ Future rows leak into training. The model **cheats**.

**Time-Based Split (RIGHT)**  
`[Jan|Feb|Mar|...|Oct|Nov|Dec]` → first 80% (past) | last 20% (future)  
→ Honest evaluation. Model trained only on the past.

**Leakage Delta** = `AUC(random split)` − `AUC(time split)`  

The more complex the model, the more it exploits the leak.

---

### 📊 Dataset

| Property              | Value                          |
|-----------------------|--------------------------------|
| **Source**            | KDD Cup 2009 — Orange Telecom |
| **Rows**              | 50,000 customers              |
| **Features**          | 230 (190 numerical + 40 categorical) |
| **Target**            | Churn (+1 = churned, -1 = stayed) |
| **Churn Rate**        | ~7.3% (12:1 class imbalance)  |
| **Avg. Missingness**  | 69.8%                         |

---

### 🏗️ Project Structure

```bash
kdd-churn-oracle-temporal-leakage/
├── data/
│   ├── orange_small_train.data
│   └── orange_small_train_churn.labels
├── outputs/
│   ├── clean_data.csv
│   ├── complexity_cliff.png
│   ├── leakage_delta_bars.png
│   └── leakage_results.csv
├── eda.py
├── clean.py
├── split.py
├── train.py
├── evaluate.py
├── visualize.py
├── app.py                  # Streamlit dashboard
└── requirements.txt

🚀 Run It Yourself

# 1. Clone the repo
git clone https://github.com/enghamza-AI/kdd-churn-oracle-temporal-leakage.git
cd kdd-churn-oracle-temporal-leakage

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download the dataset
# → https://www.kdd.org/kdd-cup/view/kdd-cup-2009/Data
# Place orange_small_train.data and orange_small_train_churn.labels into the data/ folder

# 4. Run the full pipeline
python eda.py
python clean.py
python split.py
python train.py
python evaluate.py
python visualize.py

# 5. Launch the interactive dashboard
streamlit run app.py


💡 What I Learned

Temporal leakage is one of the most common and costly mistakes in production ML.
With 69.8% missing data, smart imputation beats row deletion.
Class imbalance (12:1) makes accuracy worthless — AUC is the correct metric.
Model complexity and leakage go hand-in-hand: unlimited trees memorize both signal and leaked future data.
A good visualization communicates the danger far better than raw numbers.

⭐ Star this repo if you found the complexity cliff insightful!
Questions or ideas? Open an issue.


👤 Author
Hamza — BSAI student, self-studying AI Systems Engineering
GitHub: @enghamza-AI
