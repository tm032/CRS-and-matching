import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import pathlib

project_dir = pathlib.Path(__file__).parent.parent
raw_results_dir = project_dir / "raw_results"
summary_results_dir = project_dir / "summary_results"
figures_dir = project_dir / "figures"

from bipartite_graph import test_simple_lp, test_fb_star_graph

N = 50

result_df = pd.DataFrame(columns=["u_arrival_time", "optimal_alpha_lp", "optimal_alpha_simple_lp"])

for i in range(N+1):
    u_arrival_time = i
    simple_lp_solution = test_simple_lp(size_V=N, u_arrival_time=u_arrival_time)
    lp_solution = test_fb_star_graph(size_V=N, u_arrival_time=u_arrival_time, export_json=False)
    result_df = pd.concat([result_df, pd.DataFrame({"u_arrival_time": [u_arrival_time],
                                                    "optimal_alpha_lp": [lp_solution["alpha"]],
                                                    "optimal_alpha_simple_lp": [simple_lp_solution["alpha"]]})],
                          ignore_index=True)

result_df.to_csv(summary_results_dir / "compare_lp_formulation_V50.csv", index=False)