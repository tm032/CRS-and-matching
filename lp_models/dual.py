import math
import random
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import json

import pathlib

project_dir = pathlib.Path(__file__).parent.parent
raw_results_dir = project_dir / "raw_results"
figures_dir = project_dir / "figures"

gp.setParam('OutputFlag', 0)  # Suppress Gurobi output


class General_Primal:
    def __init__(self, size_V, arrival_distribution, arrival_order):
        self.size_V = size_V
        self.V = [i for i in range(size_V)]
        self.model = None
        self.ordered_pairs = [(i, j) for i in self.V for j in self.V if i < j]
        self.arrival_order = arrival_order
        self.arrival_distribution = arrival_distribution

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

    def build_model_with_uniform_weights(self):
        x = np.zeros((self.size_V, self.size_V))
        uniform_weight = 1 / (self.size_V - 1)
        for i in self.V:
            for j in self.V:
                if i != j:
                    x[i, j] = uniform_weight
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
        self.c_pairs = []
        for pi in range(len(self.arrival_order)):
            arrival_order = self.arrival_order[pi]
            ordered_pairs = [(arrival_order[i], arrival_order[j]) for i in range(self.size_V) for j in range(i+1, self.size_V)]
            self.c_pairs = self.c_pairs + [(pi, i, j) for (i, j) in ordered_pairs]
        
        self.c = self.model.addVars(self.c_pairs, lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="c")

    def _add_primal_constraints(self):
        for pair in self.ordered_pairs:
            rhs = gp.LinExpr()
            v, w = pair
            for pi in range(len(self.arrival_order)):
                arrival_order = self.arrival_order[pi]
                i = arrival_order.index(v)
                j = arrival_order.index(w)
                if i < j:
                    rhs += self.arrival_distribution[pi] * self.c[pi, v, w]
                else:
                    rhs += self.arrival_distribution[pi] * self.c[pi, w, v]

            self.model.addConstr(rhs >= self.alpha, name=f"optimality_{v}_{w}")
        
        for pi in range(len(self.arrival_order)):
            arrival_order = self.arrival_order[pi]
            for i in range(self.size_V):
                for j in range(i+1, self.size_V):
                    if self.x[arrival_order[i], arrival_order[j]] > 0:
                        self.model.addConstr(self.c[pi, arrival_order[i], arrival_order[j]] \
                                         <= 1 - sum(self.c[pi, arrival_order[k], arrival_order[i]] * self.x[arrival_order[k], arrival_order[i]] for k in range(i))\
                                            - sum(self.c[pi, arrival_order[i], arrival_order[k]] * self.x[arrival_order[i], arrival_order[k]] for k in range(i+1, j)), name=f"feasibility_{pi}_{arrival_order[i]}_{arrival_order[j]}")
                    else:
                        self.model.addConstr(self.c[pi, arrival_order[i], arrival_order[j]] == 1, name=f"feasibility_{pi}_{arrival_order[i]}_{arrival_order[j]}")

    def solve(self):
        gp.setParam('OutputFlag', 0)  # Suppress Gurobi output
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
            "c": {(pi, i, j): self.c[pi, i, j].X for (pi, i, j) in self.c_pairs}
        }

        json_solution = {
            "alpha": self.alpha.X,
            "c": {f"({pi},{i},{j})": self.c[pi, i, j].X for (pi, i, j) in self.c_pairs}
        }
        if as_json:
            return json.dumps(json_solution, indent=4)
        else:
            return solution
    

class General_Dual:
    def __init__(self, size_V, arrival_distribution, arrival_order):
        self.size_V = size_V
        self.V = [i for i in range(size_V)]
        self.model = None
        self.vertex_pairs = [(i, j) for i in self.V for j in self.V if i < j]
        self.arrival_order = arrival_order
        self.arrival_distribution = arrival_distribution
    
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

    def build_model_with_uniform_weights(self):
        x = np.zeros((self.size_V, self.size_V))
        uniform_weight = 1 / (self.size_V - 1)
        for i in self.V:
            for j in self.V:
                if i != j:
                    x[i, j] = uniform_weight
        self.build_model_with_custom_weights(x)

    def build_model_with_custom_weights(self, x):
        self.model = gp.Model()
        self.x = x

        self.relevant_pairs = [(i, j) for (i, j) in self.vertex_pairs if self.x[i, j] > 0]

        self._add_dual_variables()
        self._add_dual_constraints()
        self.model.setObjective(sum(self.gamma[gamma_pair] for gamma_pair in self.gamma_pairs), GRB.MINIMIZE)
        # self.model.setObjective(sum(self.gamma[pi,self.arrival_order[pi][i],self.arrival_order[pi][j]] for pi in range(len(self.arrival_order)) for (i, j) in self.vertex_pairs), GRB.MINIMIZE)

    def _add_dual_variables(self):
        self.beta = self.model.addVars(self.vertex_pairs, lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="beta")
        self.gamma_pairs = []
        for pi in range(len(self.arrival_order)):
            arrival_order = self.arrival_order[pi]
            ordered_pairs = [(arrival_order[i], arrival_order[j]) for i in range(self.size_V) for j in range(i+1, self.size_V)]
            self.gamma_pairs = self.gamma_pairs + [(pi, i, j) for (i, j) in ordered_pairs]

        self.gamma = self.model.addVars(self.gamma_pairs, lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="gamma")

    
    def _add_dual_constraints(self):
        self.model.addConstr(sum(self.beta[i,j] for (i, j) in self.relevant_pairs) == 1, name="pi_sum")
        
        for pi in range(len(self.arrival_order)):
            arrival_order = self.arrival_order[pi]
            for i in range(self.size_V):
                for j in range(i+1, self.size_V):
                    self.model.addConstr(self.gamma[pi,arrival_order[i],arrival_order[j]] + self.x[arrival_order[i],arrival_order[j]]
                                        * sum(self.gamma[pi,arrival_order[i],arrival_order[k]] + self.gamma[pi,arrival_order[j],arrival_order[k]] for k in range(j+1, self.size_V))
                                        >= self.arrival_distribution[pi] * self.beta[min(arrival_order[i],arrival_order[j]), max(arrival_order[i],arrival_order[j])], name=f"gamma_constraint_{pi}_{arrival_order[i]}_{arrival_order[j]}")
    
    def propose_beta_solution(self):
        last_elements = set()
        second_last_elements = set()
        for pi in range(len(self.arrival_order)):
            arrival_order = self.arrival_order[pi]
            last_elements.add(arrival_order[-1])
            second_last_elements.add(arrival_order[-2])

        
        n = self.size_V //2 
        
        if len(last_elements) == 1:
            i = last_elements.pop()
            j = self.arrival_order[0][0]
            obj = n / (2*n - 1)


            j = 0
            self.model.addConstr(self.beta[min(i, j), max(i, j)] >= obj, name=f"proposed_beta_solution")

            for k in range(self.size_V):
                    if k != i and k % 2 != k % 2:
                        self.model.addConstr(self.beta[min(k, j), max(k, j)] == obj / n, name=f"proposed_beta_solution_{i}_{j}")

            print(f"i = {i}, j = {j}")

        elif len(last_elements) == 2:
            q = math.pow(1 + ((n-1)/n), -2)
            i = last_elements.pop()
            j = last_elements.pop()
            if i % 2 != j % 2:
                self.model.addConstr(self.beta[min(i, j), max(i, j)] == q, name="proposed_beta_solution")

    def solve(self):
        # self.propose_beta_solution()
        gp.setParam('OutputFlag', 1)  # Suppress Gurobi output
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
            "objective_value": self.model.objVal,
            "gamma": {(pi, i, j): self.gamma[pi, i, j].X for (pi, i, j) in self.gamma_pairs},
            "beta": {(i, j): self.beta[i, j].X for (i, j) in pairs}
        }

        json_solution = {
            "objective_value": self.model.objVal,
            "gamma": {f"({pi},{i},{j})": self.gamma[pi, i, j].X for (pi, i, j) in self.gamma_pairs},
            "beta": {f"({i},{j})": self.beta[i, j].X for (i, j) in pairs}
        }

        if as_json:
            return json.dumps(json_solution, indent=4)
        else:
            return solution
        

if __name__ == "__main__":
    with open("complete_graph_results2.csv", "w") as f:
        f.write("size_V,N,optimal_value_dual\n")

    for size_V in [10, 20, 30, 40, 50, 60, 70, 80]:
        for N in [1, 2, 3, 4, 5]:
            vertices_pairs = [(i, j) for i in range(size_V) for j in range(i+1, size_V)]
            arrival_order = []

            step = size_V // N
            for i in range(0, size_V, step):
                arrival_order.append([(j + i) % size_V for j in range(size_V)])

    
            print(f"Arrival order: {arrival_order}")

            arrival_distribution = [1/len(arrival_order) for _ in range(len(arrival_order))]

            dual_model = General_Dual(size_V, arrival_distribution, arrival_order)
            dual_model.build_model_with_uniform_weights()
            optimal_value_dual = dual_model.solve()

            with open(f"dual_complete_{size_V}_{N}.json", "w") as f:
                f.write(dual_model.get_solution(as_json=True))

            with open("complete_graph_results2.csv", "a") as f:
                f.write(f"{size_V},{N},{optimal_value_dual}\n")
            

    # primal_model = General_Primal(size_V, arrival_distribution, arrival_order)
    # primal_model.build_model_with_uniform_weights()
    # optimal_value_primal = primal_model.solve()

    # dual_model = General_Dual(size_V, arrival_distribution, arrival_order)
    # dual_model.build_model_with_uniform_weights()
    # optimal_value_dual = dual_model.solve()

    
    # if abs(optimal_value_primal - optimal_value_dual) < 1e-6:
    #     print(f"Optimal value (Primal=Dual): {optimal_value_dual}")
    # else:
    #     print(f"Optimal value (Primal): {optimal_value_primal}")
    #     print(f"Optimal value (Dual): {optimal_value_dual}")
    
    # dual_solution = dual_model.get_solution()

    # with open("dual_complete.json", "w") as f:
    #     f.write(dual_model.get_solution(as_json=True))

