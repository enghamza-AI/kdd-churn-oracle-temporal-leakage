# visualize.py — KDD Churn Oracle

 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

os.makedirs('outputs', exist_ok=True)


print("Loading leakage results...")
df = pd.read_csv('outputs/leakage_results.csv')
print(df.to_string(index=False))


COLOR_RAND  = '#E74C3C'   
COLOR_TIME  = '#2ECC71'  
COLOR_DELTA = '#9B59B6'   
BG          = '#0F1520'   
GRID        = '#1A2240'   

models      = df['model'].tolist()
auc_rand    = df['auc_random'].tolist()
auc_time    = df['auc_time'].tolist()
deltas      = df['delta'].tolist()
x           = np.arange(len(models))


fig, ax = plt.subplots(figsize=(12, 6))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)


ax.plot(x, auc_rand, color=COLOR_RAND,  linewidth=2.5,
        marker='o', markersize=8, label='Random Split (Leaky)')
ax.plot(x, auc_time, color=COLOR_TIME,  linewidth=2.5,
        marker='s', markersize=8, label='Time Split (Honest)')


ax.fill_between(x, auc_time, auc_rand,
                alpha=0.15, color=COLOR_DELTA,
                label='Leakage Zone')


for i, (rand, time, delta) in enumerate(zip(auc_rand, auc_time, deltas)):
    if delta > 0.01:   
        mid = (rand + time) / 2
        ax.annotate(
            f'Δ{delta:.3f}',
            xy=(i, mid),
            fontsize=8,
            color=COLOR_DELTA,
            ha='center',
            fontweight='bold'
        )


ax.set_xticks(x)
ax.set_xticklabels(models, rotation=20, ha='right',
                   color='white', fontsize=9)
ax.set_ylabel('AUC Score', color='white', fontsize=11)
ax.set_xlabel('Model Complexity  →', color='white', fontsize=11)
ax.set_title(
    'KDD Churn Oracle — Complexity Cliff\n'
    'AUC inflation from temporal leakage grows with model complexity',
    color='white', fontsize=13, fontweight='bold', pad=15
)
ax.tick_params(colors='white')
ax.yaxis.set_tick_params(labelcolor='white')
ax.spines[['top','right','left','bottom']].set_color(GRID)
ax.grid(axis='y', color=GRID, linewidth=0.8)
ax.legend(facecolor=BG, edgecolor=GRID,
          labelcolor='white', fontsize=10)
ax.set_ylim(
    max(0, min(auc_time) - 0.05),
    min(1, max(auc_rand) + 0.05)
)

plt.tight_layout()
plt.savefig('outputs/complexity_cliff.png', dpi=150,
            bbox_inches='tight', facecolor=BG)
print("\n Saved: outputs/complexity_cliff.png")
plt.close()


fig, ax = plt.subplots(figsize=(12, 5))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)

bars = ax.bar(x, deltas, color=COLOR_DELTA,
              alpha=0.85, width=0.6, edgecolor=BG)


for bar, delta in zip(bars, deltas):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.001,
        f'{delta:.4f}',
        ha='center', va='bottom',
        color='white', fontsize=9, fontweight='bold'
    )


ax.set_xticks(x)
ax.set_xticklabels(models, rotation=20, ha='right',
                   color='white', fontsize=9)
ax.set_ylabel('Leakage Delta (AUC points stolen)',
              color='white', fontsize=11)
ax.set_xlabel('Model Complexity  →', color='white', fontsize=11)
ax.set_title(
    'Temporal Leakage Delta by Model\n'
    'How many AUC points each model gains by cheating with a random split',
    color='white', fontsize=13, fontweight='bold', pad=15
)
ax.tick_params(colors='white')
ax.yaxis.set_tick_params(labelcolor='white')
ax.spines[['top','right','left','bottom']].set_color(GRID)
ax.grid(axis='y', color=GRID, linewidth=0.8, alpha=0.5)
ax.set_ylim(0, max(deltas) * 1.25)

plt.tight_layout()
plt.savefig('outputs/leakage_delta_bars.png', dpi=150,
            bbox_inches='tight', facecolor=BG)
print(" Saved: outputs/leakage_delta_bars.png")
plt.close()

print("\nVisualization complete.")
print("Next: run app.py to launch the Streamlit dashboard")
