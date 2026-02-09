import gurobipy as gp
from gurobipy import GRB
import pandas as pd

from bipartite_graph import test_simple_lp, test_fb_star_graph

N = 10

result_df = pd.DataFrame(columns=["u_arrival_time", "optimal_alpha_lp", "optimal_alpha_simple_lp"])

for i in range(N+1):
    u_arrival_time = i
    simple_lp_solution = test_simple_lp(size_V=N, u_arrival_time=u_arrival_time)
    lp_solution = test_fb_star_graph(size_V=N, u_arrival_time=u_arrival_time, export_json=False)
    result_df = pd.concat([result_df, pd.DataFrame({"u_arrival_time": [u_arrival_time],
                                                    "optimal_alpha_lp": [lp_solution["alpha"]],
                                                    "optimal_alpha_simple_lp": [simple_lp_solution["alpha"]]})],
                          ignore_index=True)

result_df.to_csv("compare_lp_formulation_V10.csv", index=False)