import pandas as pd
import matplotlib.pyplot as plt

import pathlib
project_dir = pathlib.Path(__file__).parent.parent
raw_results_dir = project_dir / "raw_results"
summary_results_dir = project_dir / "summary_results"
figures_dir = project_dir / "figures"

N=1000

# Read the processed CSV file
df = pd.read_csv(summary_results_dir / f"compare_lp_formulation_V{N}.csv")

# Plotting
plt.figure(figsize=(10, 6))

# Exclude u_arrival_time = 0 and u_arrival_time = N for better visualization
df = df[(df["u_arrival_time"] != 0) & (df["u_arrival_time"] != N)]

plt.plot(df["u_arrival_time"], df["optimal_alpha_lp"], label="Optimal Alpha for regular LP", color='blue')
plt.plot(df["u_arrival_time"], df["optimal_alpha_simple_lp"], label="Optimal Alpha for simplified LP", color='orange')
plt.xlabel("Arrival Time of u")
plt.ylabel("Optimal Alpha")
plt.title("Optimal Alpha between different LP Formulations")
plt.legend()
plt.grid()
plt.savefig(figures_dir / f"lp_comparison_plot_V{N}.png")
plt.show()