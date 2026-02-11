import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import pathlib

project_dir = pathlib.Path(__file__).parent.parent
raw_results_dir = project_dir / "raw_results"
summary_results_dir = project_dir / "summary_results"
figures_dir = project_dir / "figures"


def continuous_lp(q, display_result=True):
    model = gp.Model("continuous_model")

    # Define variables
    gamma = model.addVar(lb=0, ub=1, vtype=GRB.CONTINUOUS, name="gamma")
    alpha_f = model.addVar(lb=0, ub=1, vtype=GRB.CONTINUOUS, name="alpha_f")
    alpha_b = model.addVar(lb=0, ub=1, vtype=GRB.CONTINUOUS, name="alpha_b")
    beta_1 = model.addVar(lb=0, ub=1, vtype=GRB.CONTINUOUS, name="beta_1")
    beta_2 = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="beta_2")
    beta_3 = model.addVar(lb=0, ub=1, vtype=GRB.CONTINUOUS, name="beta_3")
    beta_4 = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="beta_4")

    # Auxiliary variables
    l1 = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="l1")
    l2 = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="l2")
    
    B = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="B")
    C = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="C")

    # Define constraints
    model.addConstr(gamma <= 0.5 * (alpha_f + alpha_b), name="gamma_constraint_1")
    model.addConstr(gamma <= 0.5* (beta_1 + beta_3), name="gamma_constraint_2")
    model.addConstr(gamma <= 0.5 * (beta_1 - beta_2 * q + beta_3 + beta_4 * q), name="gamma_constraint_3")

    model.addConstr(alpha_f <= 1 - beta_1 * q + 0.5*beta_2*q*q - (1-q) * alpha_f, name="beta_constraint_1")

    model.addConstr(B == beta_4-beta_3+l1-l2, name="B_definition")
    model.addConstr(C == 1 - beta_3 - (1-q)*alpha_b - beta_3*q - 0.5 *beta_4*q*q - l2*q, name="C_definition")
    model.addConstr(B*B <= 2*beta_4*C, name="beta_constraint_3")

    model.addConstr(alpha_f <= 1, name="alpha_f_constraint")
    model.addConstr(alpha_b <= 1, name="alpha_b_constraint")
    model.addConstr(beta_1 - beta_2 * q >= 0, name="beta_1_beta_2_constraint")
    model.addConstr(beta_3 + beta_4 * q <= 1, name="beta_3_beta_4_constraint")

    # # Set beta_2 and beta_4 to 0 to simplify
    # model.addConstr(beta_2 == 0, name="beta_2_zero_constraint")
    # model.addConstr(beta_4 == 0, name="beta_4_zero_constraint")
    

    # Define objective
    model.setObjective(gamma, GRB.MAXIMIZE)
    model.setParam('OutputFlag', 0)  # Suppress Gurobi output

    model.optimize()

    solution = {
        "gamma": gamma.X,
        "alpha_f": alpha_f.X,
        "alpha_b": alpha_b.X,
        "beta_1": beta_1.X,
        "beta_2": beta_2.X,
        "beta_3": beta_3.X,
        "beta_4": beta_4.X
    }

    if display_result:
        print(f"Optimal gamma: {solution['gamma']}"
              f"\nOptimal alpha_f: {solution['alpha_f']}"
              f"\nOptimal alpha_b: {solution['alpha_b']}"
              f"\nOptimal beta_1: {solution['beta_1']}"
              f"\nOptimal beta_2: {solution['beta_2']}"
              f"\nOptimal beta_3: {solution['beta_3']}"
              f"\nOptimal beta_4: {solution['beta_4']}")

    return solution

def f(beta_3, beta_4, alpha_b, q, tau):
    return beta_3 + beta_4 * tau + (1-q) * alpha_b + beta_3*(q-tau) + 0.5 *beta_4 *(q*q-tau*tau)

def solve_lp_for_q_values(q_values):
    results = []
    for q in q_values:
        solution = continuous_lp(q, display_result=False)
        print(f"q: {q}, gamma: {solution['gamma']}")

        # t_star = 1 - solution["beta_3"]/solution["beta_4"] if solution["beta_4"] > 0 else 0

        results.append({"q": q, "gamma": solution["gamma"], "alpha_f":
                        solution["alpha_f"], "alpha_b": solution["alpha_b"],
                        "beta_1": solution["beta_1"], "beta_2": solution["beta_2"],
                        "beta_3": solution["beta_3"], "beta_4": solution["beta_4"],
                        # "tau=0": f(solution["beta_3"], solution["beta_4"], solution["alpha_b"], q, 0),
                        # "tau=t*": f(solution["beta_3"], solution["beta_4"], solution["alpha_b"], q, t_star),
                        # "tau=q": f(solution["beta_3"], solution["beta_4"], solution["alpha_b"], q, q),
                        # "t*": t_star
                        })                        
        
    return results

import pathlib

if __name__ == "__main__":
    precision = 0.001
    results = solve_lp_for_q_values(q_values=[precision * i + 0.5 for i in range(int(0.5/precision + 1))])
    results_df = pd.DataFrame(results)
    current_dir = pathlib.Path(__file__).parent
    results_df.to_csv(summary_results_dir / "continuous_lp_results2.csv", index=False)