class Graph:
    def __init__(self, vertices, edges):
        self.V = vertices
        self.edges = edges
        self.adjacency_list = self.build_adjacency_list()
        self.degree = {i: len(self.adjacency_list[i]) for i in range(self.V)}

    def build_adjacency_list(self):
        adjacency_list = {v: [] for v in range(self.V)}
        for v, w in self.edges:
            adjacency_list[v].append(w)
            adjacency_list[w].append(v)
        return adjacency_list

    def simplify(self):
        # Simplify the graph by removing irrelevant vertices
        change = True
        while change:
            change = False
            for v in list(self.adjacency_list.keys()):
                if self.degree[v] == 1:
                    # Remove leaf vertices
                    neighbor = self.adjacency_list[v][0]
                    self.adjacency_list[neighbor].remove(v)
                    self.adjacency_list.pop(v)
                    self.degree[neighbor] -= 1
                    self.degree.pop(v)
                    change = True
                elif self.degree[v] == 2:
                    # Merge degree two vertices
                    u, w = self.adjacency_list[v]
                    if u != w:
                        self.adjacency_list[u].remove(v)
                        self.adjacency_list[w].remove(v)
                        if u not in self.adjacency_list[w]:
                            self.adjacency_list[u].append(w)
                            self.adjacency_list[w].append(u)
                    self.adjacency_list.pop(v)
                    self.degree[u] -= 1
                    self.degree[w] -= 1
                    self.degree.pop(v)
                    change = True
        self.edges = [(v, w) for v in self.adjacency_list for w in self.adjacency_list[v] if v < w]
        self.V = len(self.adjacency_list)

def compute_lower_bound(graph):
    # Euler characteristic formula for planar graphs as a lower bound
    E = len(graph.edges)
    V = graph.V
    return max(1, (E - V + 2) // 2)

def recursive_embedding(graph, current_genus, max_genus):
    if current_genus > max_genus:
        return False
    if is_planar(graph):
        return current_genus  # Found embedding
    # More recursion to embed further
    for next_genus in range(current_genus + 1, max_genus + 1):
        if recursive_embedding(graph, next_genus, max_genus):
            return next_genus
    return False

def is_planar(graph):
    # Basic planarity check (placeholder)
    if graph.V <= 4 or len(graph.edges) <= 3 * graph.V - 6:
        return True
    return False

def main():
    vertices = 8
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
    graph = Graph(vertices, edges)
    graph.simplify()
    
    initial_genus = compute_lower_bound(graph)
    max_genus = initial_genus  # This could be adjusted based on further analysis
    
    result = recursive_embedding(graph, initial_genus, max_genus)
    print(f"Minimum genus found: {result if result else 'No embedding found within the bounds'}")

if __name__ == "__main__":
    main()
