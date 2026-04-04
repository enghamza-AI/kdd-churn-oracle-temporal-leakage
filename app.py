# app.py — KDD Churn Oracle


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os


st.set_page_config(
    page_title="KDD Churn Oracle",
    page_icon="📡",
    layout="wide"
)


st.markdown("""
<style>
    .main { background-color: #0F1520; }
    .metric-card {
        background: #141B2E;
        border: 1px solid #1A2240;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
    .big-number {
        font-size: 2rem;
        font-weight: bold;
        color: #C9A84C;
    }
    .label {
        font-size: 0.85rem;
        color: #9AA5BC;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_results():
    return pd.read_csv('outputs/leakage_results.csv')

@st.cache_data
def load_clean_data():
    return pd.read_csv('outputs/clean_data.csv')

results = load_results()
df      = load_clean_data()


st.markdown("# 📡 KDD Churn Oracle")
st.markdown(
    "**Measuring the exact cost of trusting a shuffled split "
    "— across 7 model complexities.**"
)
st.markdown("---")


st.markdown(" Dataset Overview")

total     = len(df)
churned   = int(df['churn'].sum())
stayed    = total - churned
n_features = df.shape[1] - 1

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="big-number">{total:,}</div>
        <div class="label">Total Customers</div>
    </div>""", unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="big-number">{churned:,}</div>
        <div class="label">Churned (+1)</div>
    </div>""", unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="big-number">{churned/total*100:.1f}%</div>
        <div class="label">Churn Rate</div>
    </div>""", unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="big-number">{n_features}</div>
        <div class="label">Features Used</div>
    </div>""", unsafe_allow_html=True)

st.markdown("---")


st.markdown(" What is Temporal Leakage?")

col_a, col_b = st.columns(2)

with col_a:
    st.error("""
    ** Random Split (The Wrong Way)**

    Shuffle all 50,000 rows randomly. Split 80/20.
    December customers end up in your training set.
    January customers end up in your test set.
    The model sees the future during training.
    AUC looks great. Real-world performance collapses.
    """)

with col_b:
    st.success("""
    ** Time Split (The Right Way)**

    Keep the original row order. Train on first 80%.
    Test on last 20%. The model never sees future rows.
    AUC is lower — but it is honest.
    What you measure is what you get in production.
    """)

st.markdown("---")


st.markdown(" The Complexity Cliff")
st.markdown(
    "The gap between the red line (random split) and green line "
    "(time split) is the **temporal leakage delta**. "
    "Notice how the gap grows as model complexity increases."
)

if os.path.exists('outputs/complexity_cliff.png'):
    st.image('outputs/complexity_cliff.png', use_container_width=True)
else:
    st.warning("Run visualize.py first to generate charts.")


st.markdown("## 🟣 Leakage Delta by Model")
st.markdown(
    "Each bar shows the exact number of AUC points **stolen** "
    "by using a random shuffle instead of a proper time split."
)

if os.path.exists('outputs/leakage_delta_bars.png'):
    st.image('outputs/leakage_delta_bars.png', use_container_width=True)

st.markdown("---")


st.markdown(" Full Results Table")

def color_delta(val):
    if val > 0.05:
        return 'background-color: #4a1a1a; color: #ff6b6b'
    elif val > 0.02:
        return 'background-color: #3a2a1a; color: #ffa94d'
    else:
        return 'background-color: #1a3a1a; color: #69db7c'

styled = results.style.applymap(
    color_delta, subset=['delta']
).format({
    'auc_random': '{:.4f}',
    'auc_time':   '{:.4f}',
    'delta':      '{:.4f}',
})

st.dataframe(styled, use_container_width=True)


st.markdown("---")
st.markdown(" Key Insight")

max_row = results.loc[results['delta'].idxmax()]
min_row = results.loc[results['delta'].idxmin()]
avg_delta = results['delta'].mean()

st.markdown(f"""
- **Biggest leaker:** `{max_row['model']}` stole
  **{max_row['delta']:.4f} AUC points** from the random shuffle
- **Most honest:** `{min_row['model']}` had only
  **{min_row['delta']:.4f} AUC points** of leakage
- **Average inflation** across all 7 models: **{avg_delta:.4f} AUC points**

This means if you reported the random-split AUC without checking,
your model would appear **{avg_delta:.4f} better** than it actually
performs in production. On a 12:1 imbalanced dataset with 50,000
customers, that gap costs real money.
""")

st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#6B7A94; font-size:12px;'>"
    "Built by Hamza · Stage 1 Week 3 · KDD Churn Oracle · "
    "github.com/enghamza-AI/kdd-churn-oracle-temporal-leakage"
    "</div>",
    unsafe_allow_html=True
)
