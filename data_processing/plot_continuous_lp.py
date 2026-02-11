import pandas as pd
import matplotlib.pyplot as plt
import pathlib

project_dir = pathlib.Path(__file__).parent.parent
raw_results_dir = project_dir / "raw_results"
summary_results_dir = project_dir / "summary_results"
figures_dir = project_dir / "figures"

if __name__ == "__main__":
    # Read the processed CSV file
    current_dir = pathlib.Path(__file__).parent

    continuous_df = pd.read_csv(summary_results_dir / "continuous_lp_results.csv")

    v_1000_optimal_df = pd.read_csv(summary_results_dir / "compare_lp_formulation_V1000.csv")
    v_1000_optimal_df = v_1000_optimal_df[(v_1000_optimal_df["u_arrival_time"] >= 500) & (v_1000_optimal_df["u_arrival_time"] < 1000) & (v_1000_optimal_df["u_arrival_time"] % 10 == 9)]

    v_100_optimal_df = pd.read_csv(summary_results_dir / "compare_lp_formulation_V100.csv")
    v_100_optimal_df = v_100_optimal_df[(v_100_optimal_df["u_arrival_time"] >= 50) & (v_100_optimal_df["u_arrival_time"] < 100)]


    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(continuous_df["q"], continuous_df["gamma"], label="Optimal selectability as n -> infinity", color='blue')
    plt.scatter(v_1000_optimal_df["u_arrival_time"]/1000, v_1000_optimal_df["optimal_alpha_lp"], label="Optimal selectability n = 1000 vertices", color='orange')
    plt.scatter(v_100_optimal_df["u_arrival_time"]/100, v_100_optimal_df["optimal_alpha_lp"], label="Optimal selectability n = 100 vertices", color='green')
    plt.xlabel("h/n (q)")
    plt.ylabel("Optimal Selectability")
    plt.title("Optimal Selectability for different arrival times of u")
    plt.legend()
    plt.grid()
    plt.savefig(figures_dir / "continuous_lp_gamma_plot.png")
    plt.show()