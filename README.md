# Contention Resolution Scheme in Online Bipartite Matching
Tsugunobu Miyake

This is an implementation of the LP models that finds an optimal selectability for a given distributions of arrival orders.

### Directory Structures
- `lp_models/bipartite_graph.py`: Contains the main LP Model that finds an optimal selectability for given bipartite graph and arrival order distributions.
- `continuous/cont_formulation.py`: Contains nonlinear program describing a FB-CRS LP model as `n -> infinity`.
- `data_processing`: Plotting data etc.
- `figures`: Figures that summarizes the results.
- `raw_results`: Optimal solutions stored in `.json` or `.pkl`. 
- `summary_results`: Summary of the `raw_results` in `.csv`.
