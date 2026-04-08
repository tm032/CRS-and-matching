import pandas as pd
import matplotlib.pyplot as plt

import pathlib
project_dir = pathlib.Path(__file__).parent.parent
raw_results_dir = project_dir / "raw_results"
summary_results_dir = project_dir / "summary_results"
figures_dir = project_dir / "figures"

print("Plotting complete graph results...")

df = pd.read_csv(summary_results_dir / "complete_graph_results.csv")
plt.figure(figsize=(10, 6))
for N in sorted(df["N"].unique()):
    subset = df[df["N"] == N]
    plt.plot(subset["size_V"], subset["optimal_value_dual"], marker='o', label=f"N={N}")
plt.xlabel("Size of V")
plt.ylabel("Optimal Value of Dual")
plt.title("Optimal Value of Dual vs Size of V for Complete Graphs")
plt.legend()
plt.grid()
plt.savefig(figures_dir / "complete_graph_results.png")
plt.show()