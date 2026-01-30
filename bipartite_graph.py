import math
import gurobipy as gp
from gurobipy import GRB
import json
import pickle as pkl
import random

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
        self.arrival_order_list = arrival_order_list
        
        if arrival_order_list is not None:
            for idx, vertex in enumerate(arrival_order_list):
                self.arrival_order[vertex] = idx

    def lt(self, v1, v2):
        return self.arrival_order[v1] < self.arrival_order[v2]
    
    def generate_reverse_order(self, p):
        reversed_order = list(reversed(self.arrival_order))
        return Ordering(self.bipartite_graph, p, reversed_order)
    
    def get_previous_vertex(self, vertex, same_side=False):
        idx = self.arrival_order[vertex]
        if idx == 0:
            return None
        for i in range(idx-1, -1, -1):
            prev_vertex = self.arrival_order_list[i]
            if same_side:
                if (vertex in self.bipartite_graph.U and prev_vertex in self.bipartite_graph.U) or \
                   (vertex in self.bipartite_graph.V and prev_vertex in self.bipartite_graph.V):
                    return prev_vertex
        return None
    
    def get_next_vertex(self, vertex, same_side=False):
        idx = self.arrival_order[vertex]
        if idx == len(self.arrival_order_list) - 1:
            return None
        for i in range(idx+1, len(self.arrival_order_list)):
            next_vertex = self.arrival_order_list[i]
            if same_side:
                if (vertex in self.bipartite_graph.U and next_vertex in self.bipartite_graph.U) or \
                    (vertex in self.bipartite_graph.V and next_vertex in self.bipartite_graph.V):
                      return next_vertex
        return None

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
    
    def build_lp_constraints(self, tight_constr):
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

        if tight_constr:
            for ordering in self.arrival_orders:
                for u in self.bipartite_graph.U:
                    for v in self.bipartite_graph.V:
                        prev_v = ordering.get_previous_vertex(v, same_side=True)
                        if prev_v is None: continue
                        if ordering.lt(u, v) and ordering.lt(u, prev_v):
                            self.model.addConstr(self.c[ordering.id][(u, v)] <= self.c[ordering.id][(u, prev_v)], name=f"constr_3_{ordering.id}_{u}_{v}")
                        if ordering.lt(v, u) and ordering.lt(prev_v, u):
                            self.model.addConstr(self.c[ordering.id][(v, u)] <= self.c[ordering.id][(prev_v, u)], name=f"constr_3_{ordering.id}_{u}_{v}")
                
    def build_model(self, tight_constr=False):
        self.build_lp_variables()
        self.build_lp_constraints(tight_constr)
        self.model.setObjective(self.alpha, GRB.MAXIMIZE)

    def optimize(self, suppress_output=True):
        if suppress_output:
            self.model.setParam('OutputFlag', 0)  # Suppress Gurobi output

        self.model.optimize()

        self.solution = {}
        self.json_compatible_solution = {}

        self.solution["alpha"] = self.alpha.X
        self.json_compatible_solution["alpha"] = self.alpha.X
        
        for ordering in self.arrival_orders:
            self.solution[ordering.id] = {}
            self.json_compatible_solution[ordering.id] = {}

            for u in self.bipartite_graph.U:
                for v in self.bipartite_graph.V:
                    if ordering.lt(u, v):
                        self.solution[ordering.id][(u, v)] = self.c[ordering.id][(u, v)].X
                        self.json_compatible_solution[ordering.id][f"({u},{v})"] = self.c[ordering.id][(u, v)].X
                    else:
                        self.solution[ordering.id][(v, u)] = self.c[ordering.id][(v, u)].X
                        self.json_compatible_solution[ordering.id][f"({v},{u})"] = self.c[ordering.id][(v, u)].X
    
    def get_solution(self):
        return self.solution
    
    def export_solution_to_json(self, json_file_name):
        with open(json_file_name, 'w') as f:
            json.dump(self.json_compatible_solution, f, indent=4)

    def export_solution_to_pickle(self, pickle_file_name):
        with open(pickle_file_name, 'wb') as f:
            pkl.dump(self.solution, f)

    def return_model(self):
        return self.model
        
def test_fb_star_graph(size_V, u_arrival_time, result_file_name=None, random_order=False, tight_constr=False):
    return test_bipartite_graph(size_U=1, 
                                size_V=size_V, 
                                u_arrival_times=[u_arrival_time], 
                                result_file_name=result_file_name, 
                                fb=True, 
                                random_order=random_order,
                                tight_constr=tight_constr)

def test_single_arrival_star_graph(size_V, u_arrival_time, result_file_name=None, random_order=False, tight_constr=False):
    return test_bipartite_graph(size_U=1, 
                                size_V=size_V, 
                                u_arrival_times=[u_arrival_time], 
                                result_file_name=result_file_name, 
                                fb=False, 
                                random_order=random_order, 
                                tight_constr=tight_constr)

def test_bipartite_graph(size_U, size_V, u_arrival_times, result_file_name=None, export_json=True, fb=False, random_order=False, tight_constr=False):
    if result_file_name is None:
        result_file_name = f"single_arrival_bipartite_graph_U{size_U}_V{size_V}_arrival{u_arrival_times}.json"

    bipartite_graph = BipartiteGraph(size_U, size_V)
    
    for u in bipartite_graph.U:
        for v in bipartite_graph.V:
            bipartite_graph.set_weight(u, v, 1/max(size_U, size_V)) # Uniform fractional matching weights

    if random_order:
        all_vertices = bipartite_graph.V + bipartite_graph.U
        random.shuffle(all_vertices)
        order = all_vertices
    else:
        order = [bipartite_graph.V[i] for i in range(u_arrival_times[0])] + [bipartite_graph.U[0]]
        for t in range(1, len(u_arrival_times)):
            order += [bipartite_graph.V[i] for i in range(u_arrival_times[t-1], u_arrival_times[t])] + [bipartite_graph.U[t]]
        order += [bipartite_graph.V[i] for i in range(u_arrival_times[-1], size_V)]
        
    if fb:
        reverse_order = list(reversed(order))
        ordering_1 = Ordering("ordering_1", bipartite_graph, 0.5, order)
        ordering_2 = Ordering("ordering_2", bipartite_graph, 0.5, reverse_order)
        arrival_orderings = [ordering_1, ordering_2]
    else:
        ordering = Ordering("ordering", bipartite_graph, 1.0, order)
        arrival_orderings = [ordering]

    lp_model = LP_Model(bipartite_graph, arrival_orderings)
    lp_model.build_model(tight_constr=tight_constr)
    lp_model.optimize()

    if export_json:
        lp_model.export_solution_to_json(json_file_name=result_file_name)

    model = lp_model.return_model()
    nb_reduced_cost_strictly_positive = True

    for v in model.getVars():
        if v.VBasis != GRB.BASIC:  # Nonbasic variable at lower bound
            if abs(v.RC) < 1e-6:
                nb_reduced_cost_strictly_positive = False
            # print(f"Nonbasic: {v.VarName}={v.X}, Reduced Cost={v.RC}")
        else:
            # print(f"Basic:    {v.VarName}={v.X}")
            pass

    if fb:
        print(f"FB Arrival Order, |U|={size_U}, |V|={size_V}, u_arrival_times={u_arrival_times}, alpha={lp_model.get_solution()['alpha']}, nb_reduced_cost>0: {nb_reduced_cost_strictly_positive}")
    else:
        print(f"Single Arrival Order, |U|={size_U}, |V|={size_V}, u_arrival_times={u_arrival_times}, alpha={lp_model.get_solution()['alpha']}, nb_reduced_cost>0: {nb_reduced_cost_strictly_positive}")


    return lp_model.get_solution()

def test_bipartite_graph_all_permutations(size_U, size_V, result_file_name=None):
    # Test all permutations of arrival orders for star graph with single u vertex
    import itertools
    
    permutation_size = math.factorial(size_V + size_U)

    if result_file_name is None:
        if permutation_size <= 1000:
            result_file_name = f"bipartite_all_permutations_U{size_U}_V{size_V}.json"
        else: # save in pickle file
            result_file_name = f"bipartite_all_permutations_U{size_U}_V{size_V}.pkl"
            
    bipartite_graph = BipartiteGraph(size_U, size_V)
    for u in bipartite_graph.U:
        for v in bipartite_graph.V:
            bipartite_graph.set_weight(u, v, 1/size_V) # Uniform fractional matching weights
    
    arrival_orderings = []

    permutations = itertools.permutations(bipartite_graph.V + bipartite_graph.U)

    p = 1/permutation_size
    
    for i, perm in enumerate(permutations):
        ordering = Ordering(f"ordering_{i+1}", bipartite_graph, p, list(perm))
        arrival_orderings.append(ordering)
    
    lp_model = LP_Model(bipartite_graph, arrival_orderings)
    lp_model.build_model()
    lp_model.optimize(suppress_output=False)

    if permutation_size <= 1000:
        lp_model.export_solution_to_json(result_file_name)
    else:
        lp_model.export_solution_to_pickle(result_file_name)
    
    print(f"All permutations, |U|={size_U}, |V|={size_V}, alpha={lp_model.get_solution()['alpha']}")

if __name__ == "__main__":
    # test_single_arrival_star_graph(size_V=10, u_arrival_time=0, result_file_name="test.json")
    test_fb_star_graph(size_V=1000, u_arrival_time=100, result_file_name="test1.json", tight_constr=False)
    test_fb_star_graph(size_V=1000, u_arrival_time=100, result_file_name="test2.json", tight_constr=True)

    # test_bipartite_graph(size_U=10, size_V=10, u_arrival_times=None, result_file_name="test.json", fb=True, random_order=True)

    # test_bipartite_graph(size_U=1, size_V=100, u_arrival_times=[99], result_file_name="test_bipartite.json", fb=False)

    # test_bipartite_graph(size_U=2, size_V=10, u_arrival_times=[4,8], result_file_name="single_bipartite_U2_V10_arrival4_8.json", fb=False)
    # test_bipartite_graph(size_U=2, size_V=100, u_arrival_times=[40,80], result_file_name="single_bipartite_U2_V100_arrival40_80.json", fb=False)
    # test_bipartite_graph(size_U=2, size_V=1000, u_arrival_times=[400,800], result_file_name="single_bipartite_U2_V1000_arrival400_800.json", fb=False)
    # test_bipartite_graph(size_U=2, size_V=10, u_arrival_times=[4,8], result_file_name="fb_bipartite_U2_V10_arrival4_8.json", fb=True)
    # test_bipartite_graph(size_U=2, size_V=100, u_arrival_times=[40,80], result_file_name="fb_bipartite_U2_V100_arrival40_80.json", fb=True)
    # test_bipartite_graph(size_U=2, size_V=1000, u_arrival_times=[400,800], result_file_name="fb_bipartite_U2_V1000_arrival400_800.json", fb=True)


    # test_bipartite_graph(size_U=2, size_V=100, u_arrival_times=[1, 50], result_file_name="fb_bipartite_graph_U2_V100_arrival40_80.json", fb=True)
    # test_fb_star_graph(size_V=1000, u_arrival_time=400)
    # test_fb_star_graph(size_V=100, u_arrival_time=40)
    # test_fb_star_graph(size_V=10, u_arrival_time=4)

    # test_fb_star_graph(size_V=1000, u_arrival_time=100)
    # test_fb_star_graph(size_V=100, u_arrival_time=10)
    # test_fb_star_graph(size_V=10, u_arrival_time=1)

    # test_single_arrival_star_graph(size_V=1000, u_arrival_time=400)
    # test_single_arrival_star_graph(size_V=100, u_arrival_time=40)
    # test_single_arrival_star_graph(size_V=10, u_arrival_time=4)

    # test_single_arrival_star_graph(size_V=1000, u_arrival_time=100)
    # test_single_arrival_star_graph(size_V=100, u_arrival_time=10)
    # test_single_arrival_star_graph(size_V=10, u_arrival_time=1)

    # result = {}
    # for i in range(501):
    #     result[i] = test_bipartite_graph(
    #             size_U=1, 
    #             size_V=500, 
    #             u_arrival_times=[i], 
    #             export_json=False, 
    #             fb=True)

    # json.dump(result, open("fb_bipartite_graph_U1_V500.json", "w"), indent=4)

    # selectability = []
    # for i in range(len(result)):
    #     selectability.append(result[i]['alpha'])

    # test_bipartite_graph_all_permutations(size_U=2, size_V=7)