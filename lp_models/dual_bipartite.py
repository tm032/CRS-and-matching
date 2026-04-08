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


# class FB_Primal:
#     def __init__(self, size_V):
#         self.size_V = size_V
#         self.V = [i for i in range(size_V)]
#         self.model = None
#         self.ordered_pairs = [(i, j) for i in self.V for j in self.V if i < j]


#     def build_bipartite_model(self, U, weights=None):
#         x = np.zeros((self.size_V, self.size_V))
#         uniform_weights = 1 / max(len(U), self.size_V - len(U))
#         for i in self.V:
#             for j in self.V:
#                 if i in U and j in U or i not in U and j not in U:
#                     x[i, j] = 0
#                 else:
#                     if weights is not None:
#                         x[i, j] = weights.get((i, j), uniform_weights)
#                     else:
#                         x[i, j] = uniform_weights

#         self.build_model_with_custom_weights(x)
        

#     def build_model_with_custom_weights(self, x):
#         self.model = gp.Model()
#         self.x = x

#         self.relevant_pairs = [(i, j) for (i, j) in self.ordered_pairs if self.x[i, j] > 0]

#         self._add_primal_variables()
#         self._add_primal_constraints()
#         self.model.setObjective(self.alpha, GRB.MAXIMIZE)


#     def _add_primal_variables(self):
#         self.alpha = self.model.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="alpha")
#         self.c_f = self.model.addVars(self.ordered_pairs, lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="c_f")
#         self.c_b = self.model.addVars([(j,i) for (i,j) in self.ordered_pairs], lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="c_b")
    
#     def _add_primal_constraints(self):
#         for (i, j) in self.relevant_pairs:
#             self.model.addConstr(0.5*self.c_f[i, j] + 0.5*self.c_b[j, i] >= self.alpha, name=f"optimality_{i}_{j}")
#             self.model.addConstr(self.c_f[i, j] <= 1 - sum(self.c_f[h, i] * self.x[h, i] for h in range(i)) 
#                                     - sum(self.c_f[i, h] * self.x[i, h] for h in range(i+1, j)), name=f"feasibility_{i}_{j}")
#             self.model.addConstr(self.c_b[j, i] <= 1 - sum(self.c_b[h, j] * self.x[h, j] for h in range(j+1, self.size_V)) 
#                                     - sum(self.c_b[j, h] * self.x[j, h] for h in range(i+1, j)), name=f"feasibility_{j}_{i}")
    
#     def get_primal_model(self):
#         return self.model
    
#     def solve(self):
#         self.model.optimize()
#         if self.model.status == GRB.OPTIMAL:
#             return self.model.objVal
#         else:
#             raise Exception("No optimal solution found.")
        
#     def get_solution(self, as_json=False):
#         if self.model is None:
#             raise Exception("Model has not been built yet.")
        
#         solution = {
#             "alpha": self.alpha.X,
#             "c_f": {(i, j): self.c_f[i, j].X for (i, j) in self.relevant_pairs},
#             "c_b": {(j, i): self.c_b[j, i].X for (i, j) in self.relevant_pairs}
#         }

#         json_solution = {
#             "alpha": self.alpha.X,
#             "c_f": {f"({i},{j})": self.c_f[i, j].X for (i, j) in self.relevant_pairs},
#             "c_b": {f"({j},{i})": self.c_b[j, i].X for (i, j) in self.relevant_pairs}
#         }
#         if as_json:
#             return json.dumps(json_solution, indent=4)
#         else:
#             return solution

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

            # if i % 2 != j % 2:
            #     self.model.addConstr(self.beta[min(i, j), max(i, j)] == obj, name="proposed_beta_solution")
            # else:
            #     k = 0
            #     while True:
            #         if self.arrival_order[0][k] % 2 != i % 2:
            #             break
            #         k += 1
            #     while True:
            #         if self.arrival_order[0][k] % 2 == i % 2:
            #             self.model.addConstr(self.beta[min(i, j), max(i, j)] >= obj, name=f"proposed_beta_solution_{i}_{k}")
            #             break
            #         j = self.arrival_order[0][k]
            #         k += 1

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
        self.propose_beta_solution()
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
            "gamma": {(pi, i, j): self.gamma[pi, i, j].X for (pi, i, j) in self.gamma_pairs},
            "beta": {(i, j): self.beta[i, j].X for (i, j) in pairs}
        }

        json_solution = {
            "gamma": {f"({pi},{i},{j})": self.gamma[pi, i, j].X for (pi, i, j) in self.gamma_pairs},
            "beta": {f"({i},{j})": self.beta[i, j].X for (i, j) in pairs}
        }

        if as_json:
            return json.dumps(json_solution, indent=4)
        else:
            return solution
        

if __name__ == "__main__":
    size_V = 20
    U = [2*i for i in range(size_V//2)]  # subset U

    N = 10  # number of arrival orders

    arrival_order = [
        # [i for i in range(size_V)],
        # [size_V-1 - i for i in range(size_V)],
        # [i + size_V//2 for i in range(size_V//2)] + [i for i in range(size_V//2)],
    ]

    # arrival_order = [
    #     [i for i in range(size_V)],
    #     [(i+1) % size_V for i in range(size_V)],
    #     #[(i+2) % size_V for i in range(size_V)],
    #     [(i+3) % size_V for i in range(size_V)],
    # ]    

    # # order = 

    

    arrival_order = [
        [0,2,4,6,8,10,12,14,16,18,1,3,5,7,9,11,13,15,17,19],
        [0,2,4,6,8,10,12,14,16,18,19,17,15,13,11,9,7,5,3,1],
        # [0,2,4,6,8,10,12,14,16,18,1,17,15,13,11,9,7,5,3,19],
        # [0,2,4,6,8,10,12,14,16,18,9,7,5,3,1,11,17,15,13,19],
        # [1,3,5,7,9,11,13,15,17,0,2,4,6,8,10,12,14,16,19,18],
        # [1,2,0,4,3,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19],
        # [1,2,0,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19],
        # # [2,1,0,3,4,5,6,7,8,9,10,11,12,13,14,15,16,19,18,17],
        # [4,1,0,3,2,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19],
        # [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,0,19],
        # [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,0,1,19],
        # # [18,19,16,17,14,15,12,13,10,11,8,9,6,7,4,5,2,3,0,1],
        # # [18,19,16,17,14,15,12,13,10,11,8,9,6,7,4,5,2,1,0,3],
        # # [18,19,16,17,14,15,12,13,10,11,8,9,6,7,4,3,2,1,0,5],
        # # # [18,19,16,17,14,15,12,13,10,11,8,9,6,7,4,3,5,1,0,2],

        # # # [18,19,16,17,14,15,12,13,10,11,8,9,6,7,4,5,2,0,3,1],
        # [1,0,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,18,17,19],
        # [18,0,16,17,14,15,12,13,10,11,8,9,6,7,4,5,2,3,1,19],

        # [0,2,4,6,8,10,12,14,16,18,1,3,5,7,9,11,13,15,17,19],
        # [18,16,14,12,10,8,6,4,2,0,17,15,13,11,9,7,5,3,1,19],
    ]

    # arrival_order = [
    #     [0,2,4,6,8,10,12,14,16,18,1,3,5,7,9,11,13,15,17,19],
    #     # [18,16,14,12,10,8,6,4,2,0,17,15,13,11,9,7,5,3,1,19],
    #     [18,16,14,12,10,8,6,4,2,0,19,17,15,13,11,9,7,5,3,1],
    # ]

    # arrival_order = [
    #     [2 * i for i in range(size_V//2)] + [2 * i + 1 for i in range(size_V//2)],
    #     [size_V - 2 * i - 2 for i in range(size_V//2)] + [size_V - 2 * i - 1 for i in range(size_V//2)],
    # ]

    # pre_order = [[i for i in range(size_V - 1)] for _ in range(N)]
    # for arr in pre_order:
    #     while True:
    #         random.shuffle(arr)
    #         if arr[0] % 2 == 1:
    #             break
        
    
    # arrival_order = [pre_order[i] + [size_V - 1] for i in range(len(pre_order))] + arrival_order

    # arrival_order = [
    #     [(i + j) % size_V for i in range(size_V)] for j in range(N)
    # ]

    mid_array1 = [[i for i in range(size_V-1)] for _ in range(N)]
    for arr in mid_array1:
        random.shuffle(arr)

    last_element_2 = 4
    
    mid_array2 = [[i for i in range(0, size_V) if i != last_element_2] for _ in range(N)]
    for arr in mid_array2:
        random.shuffle(arr)

    second_last_element_3 = 18
    mid_array3 = [[i for i in range(0, size_V) if i != second_last_element_3] for _ in range(N)]
    for arr in mid_array3:
        while True:
            random.shuffle(arr)
            if arr[-1] % 2 != second_last_element_3 % 2:
                break

    # arrival_order = [[i for i in range(size_V)]]  # single arrival order (identity)

    # arrival_order = [
    #     mid_array1[i] + [size_V-1] for i in range(len(mid_array1))
    # ] + [
    #     mid_array2[i] + [last_element_2] for i in range(len(mid_array2))
    # ] 

    # arrival_order = [
    #     mid_array1[i] + [size_V-1] for i in range(len(mid_array1))
    # ] + [
    #     mid_array3[i][:-1] + [second_last_element_3] + [mid_array3[i][-1]] for i in range(len(mid_array3))
    # ]

    # U = [i for i in range(size_V//2)]  # subset U
    # order = [i for i in range(size_V)]
    # split = size_V // 4
    # order = order[split:] + order[:split]  # rotate the order by size_V//5
    # # random.shuffle(order)

    # arrival_order = [
    #     order, list(reversed(order))
    # ]

    # arrival_arrays = [[i for i in range(size_V)] for _ in range(N)]
    # for arr in arrival_arrays:
    #     while True:
    #         random.shuffle(arr)
    #         if arr[0] % 2 == 0 and arr[-1] % 2 == 1 or arr[0] % 2 == 1 and arr[-1] % 2 == 0:
    #             break
    
    
    
    arrival_distribution = [1/len(arrival_order) for _ in arrival_order]

    primal_model = General_Primal(size_V, arrival_distribution, arrival_order)
    primal_model.build_bipartite_model(U)
    optimal_value_primal = primal_model.solve()
    # with open("general_primal_solution_200.json", "w") as f:
    #     f.write(primal_model.get_solution(as_json=True))
    

    dual_model = General_Dual(size_V, arrival_distribution, arrival_order)
    dual_model.build_bipartite_model(U)
    optimal_value_dual = dual_model.solve()

    for arr in arrival_order:
        print(arr)

    if abs(optimal_value_primal - optimal_value_dual) < 1e-6:
        print(f"Optimal value (Primal=Dual): {optimal_value_dual}")
    else:
        print(f"Optimal value (Primal): {optimal_value_primal}")
        print(f"Optimal value (Dual): {optimal_value_dual}")

    
    dual_solution = dual_model.get_solution()

    with open("general_dual_solution_200.json", "w") as f:
        f.write(dual_model.get_solution(as_json=True))

