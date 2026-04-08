# Plot duplicating_vertices_one_side.json and duplicating_vertices_distributed.json

import json
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
project_dir = Path(__file__).parent.parent
with open(project_dir / "raw_results" / "split_v_alphas.json", "r") as f:
    alphas = json.load(f)

one_side_u_0 = alphas["one_side_U_0"]
one_side_u_half = alphas["half"]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))



for size_U in range(1, 5):
    one_side_u_0_alphas = one_side_u_0[str(size_U)]
    one_side_u_half_alphas = one_side_u_half[str(size_U)]
    
    axes[0].plot([2 ** i for i in range(len(one_side_u_0_alphas))], one_side_u_0_alphas, label=f"|U|={size_U}", marker='o')
    axes[1].plot([2 ** i for i in range(len(one_side_u_half_alphas))], one_side_u_half_alphas, label=f"|U|={size_U}", marker='^')
    axes[0].set_xscale("log", base=2)
    axes[0].set_xlabel("|V|")
    axes[0].set_ylabel("Optimal Alpha")
    axes[0].set_title("Duptlication at U arrival time = 0")
    axes[0].legend()
    axes[0].grid()
    axes[1].set_xscale("log", base=2)
    axes[1].set_xlabel("|V|")
    axes[1].set_ylabel("Optimal Alpha")
    axes[1].set_title("Duplication at U arrival time = |V|/2")
    axes[1].legend()
    axes[1].grid()

plt.tight_layout()
plt.savefig(project_dir / "figures" / "duplicating_vertices_comparison2.png")
plt.show()

#     axes[0, 0].plot([2 ** i for i in range(len(one_side_u_0_alphas))], one_side_u_0_alphas, label=f"|U|={size_U}", marker='o')
#     axes[0, 1].plot([2 ** i for i in range(len(one_side_u_1_alphas))], one_side_u_1_alphas, label=f"|U|={size_U}", marker='s')
#     axes[1, 0].plot([2 ** i for i in range(len(one_side_u_half_alphas))], one_side_u_half_alphas, label=f"|U|={size_U}", marker='^')
#     axes[1, 1].plot([2 ** (i+3) for i in range(len(distributed_alphas))], distributed_alphas, label=f"|U|={size_U}", marker='D')
#     axes[0, 0].set_xscale("log", base=2)
#     axes[0, 0].set_xlabel("|V|")
#     axes[0, 0].set_ylabel("Optimal Alpha")
#     axes[0, 0].set_title("One Side Duplication (U arrival time = 0)")
#     axes[0, 0].legend()
#     axes[0, 0].grid()

#     axes[0, 1].set_xscale("log", base=2)
#     axes[0, 1].set_xlabel("|V|")
#     axes[0, 1].set_ylabel("Optimal Alpha")
#     axes[0, 1].set_title("One Side Duplication (U arrival time = 1)")
#     axes[0, 1].legend()
#     axes[0, 1].grid()
    
#     axes[1, 0].set_xscale("log", base=2)
#     axes[1, 0].set_xlabel("|V|")
#     axes[1, 0].set_ylabel("Optimal Alpha")
#     axes[1, 0].set_title("One Side Duplication (U arrival time = |V|/2)")
#     axes[1, 0].legend()
#     axes[1, 0].grid()

#     axes[1, 1].set_xscale("log", base=2)
#     axes[1, 1].set_xlabel("|V|")
#     axes[1, 1].set_ylabel("Optimal Alpha")
#     axes[1, 1].set_title("Distributed Duplication")
#     axes[1, 1].legend()
#     axes[1, 1].grid()

# plt.tight_layout()
# plt.savefig(project_dir / "figures" / "duplicating_vertices_comparison.png")
# plt.show()

#     # fig, axes = plt.subplots(1, 2, figsize=(14, 5))

#     # for size_U in range(1, 6):
#     #     one_side_alphas = one_side_data[str(size_U)]
#     #     distributed_alphas = distributed_data[str(size_U)]
#     #     size_V_one_side = [2 ** i for i in range(len(one_side_alphas))]
#     #     size_V_distributed = [2 ** i for i in range(len(distributed_alphas))]

#     #     axes[0].plot(size_V_one_side, one_side_alphas, label=f"|U|={size_U}", marker='o')
#     #     axes[1].plot(size_V_distributed, distributed_alphas, label=f"|U|={size_U}", marker='s')

#     # axes[0].set_xscale("log", base=2)
#     # axes[0].set_xlabel("|V|")
#     # axes[0].set_ylabel("Optimal Alpha")
#     # axes[0].set_title("One Side Duplication")
#     # axes[0].legend()
#     # axes[0].grid()

#     # axes[1].set_xscale("log", base=2)
#     # axes[1].set_xlabel("|V|")
#     # axes[1].set_ylabel("Optimal Alpha")
#     # axes[1].set_title("Distributed Duplication")
#     # axes[1].legend()
#     # axes[1].grid()

#     # plt.tight_layout()
#     # # plt.savefig(project_dir / "figures" / "duplicating_vertices_comparison.png")
#     # plt.show()

