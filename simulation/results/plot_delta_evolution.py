"""Plot simulation delta evolution with true value reference lines."""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'legend.fontsize': 10,
    'figure.dpi': 150,
})

df = pd.read_csv("sim_results.csv")
df['date'] = pd.to_datetime(df['date'])

true_delta = {'delta_vol': 0.5, 'delta_tail': 0.3, 'delta_ret': 0.2}
colors = {'delta_vol': '#E74C3C', 'delta_tail': '#2980B9', 'delta_ret': '#27AE60'}
labels = {'delta_vol': 'Volatility ($\\delta_1$)', 'delta_tail': 'Tail Risk ($\\delta_2$)', 'delta_ret': 'Returns ($\\delta_3$)'}

fig, ax = plt.subplots(figsize=(14, 6))

for col in ['delta_vol', 'delta_tail', 'delta_ret']:
    ax.plot(df['date'], df[col], '-o', color=colors[col], label=labels[col],
            markersize=4, linewidth=1.5, alpha=0.85)
    ax.axhline(y=true_delta[col], color=colors[col], linestyle='--', alpha=0.5, linewidth=1.2)
    ax.text(df['date'].iloc[-1] + pd.Timedelta(days=30), true_delta[col],
            f'True={true_delta[col]:.1f}', color=colors[col], fontsize=9, va='center')

ax.set_xlabel('Date')
ax.set_ylabel('Delta Weight')
ax.set_title('Simulation: Delta Evolution over Rolling Windows\n'
             '(True: $\\delta_{vol}$=0.5, $\\delta_{tail}$=0.3, $\\delta_{ret}$=0.2)')
ax.legend(loc='upper right')
ax.set_ylim(-0.05, 1.05)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.xticks(rotation=45)
ax.grid(True, alpha=0.3)

ax.axhspan(0, 0, color='gray', alpha=0.1)

mean_vol = df['delta_vol'].mean()
mean_tail = df['delta_tail'].mean()
mean_ret = df['delta_ret'].mean()
textstr = (f'Mean: $\\delta_{{vol}}$={mean_vol:.3f}, '
           f'$\\delta_{{tail}}$={mean_tail:.3f}, '
           f'$\\delta_{{ret}}$={mean_ret:.3f}')
ax.text(0.02, 0.02, textstr, transform=ax.transAxes, fontsize=9,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('delta_evolution_simulation.png', dpi=150, bbox_inches='tight')
print("Saved: delta_evolution_simulation.png")
plt.close()

print(f"\nSummary statistics:")
print(f"  delta_vol:  mean={mean_vol:.3f}, std={df['delta_vol'].std():.3f}, true=0.500, bias={mean_vol-0.5:+.3f}")
print(f"  delta_tail: mean={mean_tail:.3f}, std={df['delta_tail'].std():.3f}, true=0.300, bias={mean_tail-0.3:+.3f}")
print(f"  delta_ret:  mean={mean_ret:.3f}, std={df['delta_ret'].std():.3f}, true=0.200, bias={mean_ret-0.2:+.3f}")
