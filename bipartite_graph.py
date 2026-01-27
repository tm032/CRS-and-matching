import gurobipy as gp
from gurobipy import GRB
import json

from matplotlib.pylab import rint

class BipartiteGraph:
    def __init__(self, size_U, size_V, weights=None):
        self.U = [f"u_{i}" for i in range(1, size_U+1)]  # Set of vertices in partition U
        self.V = [f"v_{i}" for i in range(1, size_V+1)]  # Set of vertices in partition V
        self.weights = weights if weights is not None else {}

    def set_weight(self, u, v, weight):
        self.weights[(u, v)] = weight

    def get_weight(self, u, v):
        if (u, v) in self.weights:
            return self.weights[(u, v)]
        elif (v, u) in self.weights:
            return self.weights[(v, u)]
        else:
            raise ValueError(f"Weight for edge ({u}, {v}) not found.")
        
class Ordering:
    def __init__(self, ordering_id, bipartite_graph:BipartiteGraph, p, arrival_order_list:list):
        self.id = ordering_id
        self.bipartite_graph = bipartite_graph
        self.arrival_order = {}
        self.p = p          # Probability of this ordering
        
        for idx, vertex in enumerate(arrival_order_list):
            self.arrival_order[vertex] = idx
    
    def lt(self, v1, v2):
        return self.arrival_order[v1] < self.arrival_order[v2]
    
    def generate_reverse_order(self, p):
        reversed_order = list(reversed(self.arrival_order))
        return Ordering(self.bipartite_graph, p, reversed_order)
    

class LP_Model:
    def __init__(self, bipartite_graph, arrival_ordering:list[Ordering]):
        self.bipartite_graph = bipartite_graph
        self.arrival_orders = arrival_ordering
        self.model = gp.Model()
        self.c = {}

    def build_lp_variables(self):
        self.alpha = self.model.addVar(name="alpha", lb=0, vtype=GRB.CONTINUOUS)

        for ordering in self.arrival_orders:
            vertex_pairs = []
            for u in ordering.bipartite_graph.U:
                for v in ordering.bipartite_graph.V:
                    if ordering.lt(u, v):
                        vertex_pairs.append((u, v))
                    if ordering.lt(v, u):
                        vertex_pairs.append((v, u))

            self.c[ordering.id] = self.model.addVars(vertex_pairs, name=f"c_{ordering.id}", lb=0, ub=1, vtype=GRB.CONTINUOUS)
    
    def build_lp_constraints(self):
        for u in self.bipartite_graph.U:
            for v in self.bipartite_graph.V:
                constraint_expr = gp.LinExpr()

                for ordering in self.arrival_orders:
                    if ordering.lt(u, v):
                        constraint_expr += self.c[ordering.id][(u, v)] * ordering.p

                    if ordering.lt(v, u):
                        constraint_expr += self.c[ordering.id][(v, u)] * ordering.p

                self.model.addConstr(constraint_expr >= self.alpha, name=f"constr_1_{u}_{v}")

        for ordering in self.arrival_orders:
            for u in self.bipartite_graph.U:
                for v in self.bipartite_graph.V:
                    expr_1 = gp.LinExpr()
                    expr_2 = gp.LinExpr()

                    for w in self.bipartite_graph.V:
                        if ordering.lt(u, v):
                            if ordering.lt(w, u):
                                expr_1 += self.c[ordering.id][(w, u)] * self.bipartite_graph.get_weight(w, u)

                            if ordering.lt(u, w) and ordering.lt(w, v):
                                expr_2 += self.c[ordering.id][(u, w)] * self.bipartite_graph.get_weight(u, w)
                    
                    if ordering.lt(u, v):
                        self.model.addConstr(self.c[ordering.id][(u, v)] <= 1 - expr_1 - expr_2, name=f"constr_2_{ordering.id}_{u}_{v}")
                    
    def build_model(self):
        self.build_lp_variables()
        self.build_lp_constraints()
        self.model.setObjective(self.alpha, GRB.MAXIMIZE)

    def optimize(self, suppress_output=True):
        if suppress_output:
            self.model.setParam('OutputFlag', 0)  # Suppress Gurobi output

        self.model.optimize()

    def get_solution(self, export_json=False, json_file_name=None):
        solution = {}
        json_solution = {}

        solution["alpha"] = self.alpha.X
        json_solution["alpha"] = self.alpha.X

        for ordering in self.arrival_orders:
            solution[ordering.id] = {}
            json_solution[ordering.id] = {}

            for u in self.bipartite_graph.U:
                for v in self.bipartite_graph.V:
                    if ordering.lt(u, v):
                        solution[ordering.id][(u, v)] = self.c[ordering.id][(u, v)].X
                        json_solution[ordering.id][f"{u},{v}"] = self.c[ordering.id][(u, v)].X
                    else:
                        solution[ordering.id][(v, u)] = self.c[ordering.id][(v, u)].X
                        json_solution[ordering.id][f"{v},{u}"] = self.c[ordering.id][(v, u)].X

        if export_json:
            if json_file_name is None:
                json_file_name = "lp_solution.json"

            with open(json_file_name, "w") as f:
                json.dump(json_solution, f, indent=4)
        return solution
        
def test_fb_star_graph(size_V, u_arrival_time, result_file_name=None):
    if result_file_name is None:
        result_file_name = f"fb_star_graph_V{size_V}_arrival{u_arrival_time}.json"

    size_U = 1  # Number of center vertices in the star graph

    bipartite_graph = BipartiteGraph(size_U, size_V)
    
    for u in bipartite_graph.U:
        for v in bipartite_graph.V:
            bipartite_graph.set_weight(u, v, 1/size_V) # Uniform fractional matching weights

    order_1 = [bipartite_graph.V[i] for i in range(u_arrival_time)] + [bipartite_graph.U[0]] + [bipartite_graph.V[i] for i in range(u_arrival_time, size_V)]
    order_2 = list(reversed(order_1))

    ordering_1 = Ordering("ordering_1", bipartite_graph, 0.5, order_1)
    ordering_2 = Ordering("ordering_2", bipartite_graph, 0.5, order_2)

    arrival_orderings = [ordering_1, ordering_2]

    lp_model = LP_Model(bipartite_graph, arrival_orderings)
    lp_model.build_model()
    lp_model.optimize()
    solution = lp_model.get_solution(export_json=True, json_file_name=result_file_name)
    
    print(f"FB Arrival Order, |U|={size_U}, |V|={size_V}, u_arrival_time={u_arrival_time}, alpha={solution['alpha']}")

    return solution

def test_single_arrival_star_graph(size_V, u_arrival_time, result_file_name=None):
    if result_file_name is None:
        result_file_name = f"single_arrival_star_graph_V{size_V}_arrival{u_arrival_time}.json"

    size_U = 1  # Number of center vertices in the star graph

    bipartite_graph = BipartiteGraph(size_U, size_V)
    
    for u in bipartite_graph.U:
        for v in bipartite_graph.V:
            bipartite_graph.set_weight(u, v, 1/size_V) # Uniform fractional matching weights

    order = [bipartite_graph.V[i] for i in range(u_arrival_time)] + [bipartite_graph.U[0]] + [bipartite_graph.V[i] for i in range(u_arrival_time, size_V)]

    ordering = Ordering("ordering", bipartite_graph, 1.0, order)

    arrival_orderings = [ordering]

    lp_model = LP_Model(bipartite_graph, arrival_orderings)
    lp_model.build_model()
    lp_model.optimize()
    solution = lp_model.get_solution(export_json=True, json_file_name=result_file_name)

    print(f"Single Arrival Order, |U|={size_U}, |V|={size_V}, u_arrival_time={u_arrival_time}, alpha={solution['alpha']}")

    return solution


if __name__ == "__main__":
    test_fb_star_graph(size_V=1000, u_arrival_time=400)
    test_fb_star_graph(size_V=100, u_arrival_time=40)
    test_fb_star_graph(size_V=10, u_arrival_time=4)

    test_fb_star_graph(size_V=1000, u_arrival_time=100)
    test_fb_star_graph(size_V=100, u_arrival_time=10)
    test_fb_star_graph(size_V=10, u_arrival_time=1)

    test_single_arrival_star_graph(size_V=1000, u_arrival_time=400)
    test_single_arrival_star_graph(size_V=100, u_arrival_time=40)
    test_single_arrival_star_graph(size_V=10, u_arrival_time=4)

    test_single_arrival_star_graph(size_V=1000, u_arrival_time=100)
    test_single_arrival_star_graph(size_V=100, u_arrival_time=10)
    test_single_arrival_star_graph(size_V=10, u_arrival_time=1)