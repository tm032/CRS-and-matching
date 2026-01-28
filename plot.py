import json

import matplotlib.pyplot as plt

# Read the JSON file
with open('fb_bipartite_graph_U1_V500.json', 'r') as f:
    data = json.load(f)

# Extract index and alpha values
indices = list(map(int, data.keys()))
alphas = [item['alpha'] for item in data.values()]

# Plot
plt.figure(figsize=(10, 6))
plt.plot(indices, alphas, marker='o', linestyle='-', linewidth=2)
plt.xlabel('u_arrival_time')
plt.ylabel('alpha')
plt.title('u_arrival_time vs alpha')
plt.grid(True)
plt.tight_layout()
plt.savefig('u_arrival_time_vs_alpha_500.png')
plt.show()