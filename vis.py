import networkx as nx

# Create a graph object
G = nx.Graph()

# Add some edges
edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
G.add_edges_from(edges)

# Display basic information about the graph
print("Number of nodes:", G.number_of_nodes())
print("Number of edges:", G.number_of_edges())
print("Nodes:", list(G.nodes))
print("Edges:", list(G.edges))

# Draw the graph
import matplotlib.pyplot as plt

nx.draw(G, with_labels=True, font_weight='bold')
plt.show()
