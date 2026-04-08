import math

import gurobipy as gp
from gurobipy import GRB
import numpy as np
import json

import pathlib

project_dir = pathlib.Path(__file__).parent.parent
raw_results_dir = project_dir / "raw_results"
figures_dir = project_dir / "figures"


class FB_Primal:
    def __init__(self, size_V):
        self.size_V = size_V
        self.V = [i for i in range(size_V)]
        self.model = None
        self.ordered_pairs = [(i, j) for i in self.V for j in self.V if i < j]


    def build_bipartite_model(self, U, weights=None):
        x = np.zeros((self.size_V, self.size_V))
        uniform_weights = 1 / max(len(U), self.size_V - len(U))
        for i in self.V:
            for j in self.V:
                if i in U and j in U or i not in U and j not in U:
                    x[i, j] = 0
                else:
                    if weights is not None:
                        x[i, j] = weights.get((i, j), uniform_weights)
                    else:
                        x[i, j] = uniform_weights

        self.build_model_with_custom_weights(x)
        

    def build_model_with_custom_weights(self, x):
        self.model = gp.Model()
        self.x = x

        self.relevant_pairs = [(i, j) for (i, j) in self.ordered_pairs if self.x[i, j] > 0]

        self._add_primal_variables()
        self._add_primal_constraints()
        self.model.setObjective(self.alpha, GRB.MAXIMIZE)


    def _add_primal_variables(self):
        self.alpha = self.model.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="alpha")
        self.c_f = self.model.addVars(self.ordered_pairs, lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="c_f")
        self.c_b = self.model.addVars([(j,i) for (i,j) in self.ordered_pairs], lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="c_b")
    
    def _add_primal_constraints(self):
        for (i, j) in self.relevant_pairs:
            self.model.addConstr(0.5*self.c_f[i, j] + 0.5*self.c_b[j, i] >= self.alpha, name=f"optimality_{i}_{j}")
            self.model.addConstr(self.c_f[i, j] <= 1 - sum(self.c_f[h, i] * self.x[h, i] for h in range(i)) 
                                    - sum(self.c_f[i, h] * self.x[i, h] for h in range(i+1, j)), name=f"feasibility_{i}_{j}")
            self.model.addConstr(self.c_b[j, i] <= 1 - sum(self.c_b[h, j] * self.x[h, j] for h in range(j+1, self.size_V)) 
                                    - sum(self.c_b[j, h] * self.x[j, h] for h in range(i+1, j)), name=f"feasibility_{j}_{i}")
    
    def get_primal_model(self):
        return self.model
    
    def solve(self):
        self.model.optimize()
        if self.model.status == GRB.OPTIMAL:
            return self.model.objVal
        else:
            raise Exception("No optimal solution found.")
        
    def get_solution(self, as_json=False):
        if self.model is None:
            raise Exception("Model has not been built yet.")
        
        solution = {
            "alpha": self.alpha.X,
            "c_f": {(i, j): self.c_f[i, j].X for (i, j) in self.relevant_pairs},
            "c_b": {(j, i): self.c_b[j, i].X for (i, j) in self.relevant_pairs}
        }

        json_solution = {
            "alpha": self.alpha.X,
            "c_f": {f"({i},{j})": self.c_f[i, j].X for (i, j) in self.relevant_pairs},
            "c_b": {f"({j},{i})": self.c_b[j, i].X for (i, j) in self.relevant_pairs}
        }
        if as_json:
            return json.dumps(json_solution, indent=4)
        else:
            return solution
        
class FB_Dual:
    def __init__(self, size_V):
        self.size_V = size_V
        self.V = [i for i in range(size_V)]
        self.model = None
        self.ordered_pairs = [(i, j) for i in self.V for j in self.V if i < j]
    
    def build_bipartite_model(self, U, weights=None):
        x = np.zeros((self.size_V, self.size_V))
        uniform_weights = 1 / max(len(U), self.size_V - len(U))
        for i in self.V:
            for j in self.V:
                if (i in U and j in U) or (i not in U and j not in U):
                    x[i, j] = 0
                else:
                    if weights is not None:
                        x[i, j] = weights.get((i, j), uniform_weights)
                    else:
                        x[i, j] = uniform_weights

        self.build_model_with_custom_weights(x)

    def build_model_with_custom_weights(self, x):
        self.model = gp.Model()
        self.x = x

        self.relevant_pairs = [(i, j) for (i, j) in self.ordered_pairs if self.x[i, j] > 0]

        self._add_dual_variables()
        self._add_dual_constraints()
        self.model.setObjective(sum(self.gamma[i,j] + self.gamma[j,i] for (i, j) in self.ordered_pairs), GRB.MINIMIZE)

    def _add_dual_variables(self):
        self.gamma = self.model.addVars(self.ordered_pairs + [(j, i) for (i, j) in self.ordered_pairs], lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="gamma")
        self.pi = self.model.addVars(self.ordered_pairs, lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="pi")
    
    def _add_dual_constraints(self):
        a = self.model.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="a")
        b = self.model.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="b")

        self.model.addConstr(sum(self.pi[i,j] for (i, j) in self.relevant_pairs) == 1, name="pi_sum")
        for (i, j) in self.ordered_pairs:
            self.model.addConstr(self.gamma[i,j] + self.x[i,j] 
                                 * sum(self.gamma[i,k] + self.gamma[j,k] for k in range(j+1, self.size_V)) 
                                 >= 0.5 * self.pi[i,j], name=f"gamma_constraint_{i}_{j}")
            self.model.addConstr(self.gamma[j,i] + self.x[j,i] 
                                 * sum(self.gamma[i,k] + self.gamma[j,k] for k in range(i))
                                    >= 0.5 * self.pi[i,j], name=f"gamma_constraint_{j}_{i}")
            
            # if (i, j) in self.relevant_pairs:
            #     # self.model.addConstr(self.gamma[i,j] == a, name=f"gamma_a_constraint_{i}_{j}")
            #     # self.model.addConstr(self.pi[i,j] == b, name=f"pi_b_constraint_{i}_{j}")

            # if (j,i) in self.relevant_pairs:
            #     # self.model.addConstr(self.gamma[j,i] == b, name=f"gamma_a_constraint_{j}_{i}")
                
        
    def solve(self):
        self.model.optimize()
        if self.model.status == GRB.OPTIMAL:
            return self.model.objVal
        else:
            raise Exception("No optimal solution found.")

    def get_solution(self, as_json=False):
        if self.model is None:
            raise Exception("Model has not been built yet.")

        pairs = self.relevant_pairs

        solution = {
            "gamma": {(i, j): self.gamma[i, j].X for (i, j) in pairs + [(j, i) for (i, j) in self.relevant_pairs]},
            "pi": {(i, j): self.pi[i, j].X for (i, j) in pairs}   
        }

        json_solution = {
            "gamma": {f"({i},{j})": self.gamma[i, j].X for (i, j) in pairs + [(j, i) for (i, j) in self.relevant_pairs]},
            "pi": {f"({i},{j})": self.pi[i, j].X for (i, j) in pairs}   
        }

        if as_json:
            return json.dumps(json_solution, indent=4)
        else:
            return solution
        

if __name__ == "__main__":
    # size_V = 101
    # U = [2]  # subset U
    # primal_model = FB_Primal(size_V)
    # primal_model.build_bipartite_model(U)
    # optimal_value = primal_model.solve()
    # print(f"Optimal value: {optimal_value}")
    # solution = primal_model.get_solution()

    # with open("primal_solution.json", "w") as f:
    #     f.write(primal_model.get_solution(as_json=True))

    size_V = 20
    U = [0] + [i + 4 for i in range(math.ceil(size_V/3)-1)]  # subset U

    # U = [0,4,7,10,11,14,15,16,17,18]  # subset U
    # U = [3*i for i in range(math.ceil(size_V/3))]  # subset U
    # U = [2*i for i in range(size_V//2)]  # subset U

    print(U)
    # U = [i + 5 for i in range(size_V//2)]


    dual_model = FB_Dual(size_V)
    dual_model.build_bipartite_model(U)
    optimal_value_dual = dual_model.solve()
    print(f"Optimal value (Dual): {optimal_value_dual}")
    dual_solution = dual_model.get_solution()
    with open("dual_solution_10c.json", "w") as f:
        f.write(dual_model.get_solution(as_json=True))

    # print(f"primal==dual: {abs(optimal_value - optimal_value_dual) < 1e-6}")

    # size_V = [1,2,4,8,16,32,64,128,256,512,1024]
    # size_U = [1,2,3,4]
    # alphas = {"one_side_U_0": {}, "half": {}}

    # for u_size in size_U:
    #     alphas["one_side_U_0"][u_size] = []
    #     alphas["half"][u_size] = []
    #     for v_size in size_V:
    #         print(f"Testing size_V={v_size}, size_U={u_size}")
    #         U = [i for i in range(u_size)]  # subset U

    #         total_size = v_size + u_size

    #         one_side_primal_model = FB_Primal(total_size)
    #         one_side_primal_model.build_bipartite_model(U)
    #         optimal_value = one_side_primal_model.solve()
    #         print(f"Optimal value (One Side): {optimal_value}")
    #         alphas["one_side_U_0"][u_size].append(optimal_value)

    #         U = [i for i in range(v_size//2, v_size//2+u_size)]  # subset U
    #         half_primal_model = FB_Primal(total_size)
    #         half_primal_model.build_bipartite_model(U)
    #         optimal_value = half_primal_model.solve()
    #         print(f"Optimal value (Half): {optimal_value}")

    #         alphas["half"][u_size].append(optimal_value)

    # with open(raw_results_dir / "split_v_alphas.json", "w") as f:
    #     json.dump(alphas, f, indent=4)

