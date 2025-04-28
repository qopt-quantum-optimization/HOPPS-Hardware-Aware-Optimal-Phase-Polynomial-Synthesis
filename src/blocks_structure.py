from collections import deque, defaultdict

class DependencyGraph:
    def __init__(self):
        self.graph = defaultdict(list)
        self.in_degree = defaultdict(int)  # Keeps track of incoming edges

    def add_dependency(self, dependency, task):
        """dependency -> task (dependency must come before task)"""
        self.graph[dependency].append(task)
        self.in_degree[task] += 1
        if task not in self.graph:  # Ensure all nodes appear in graph
            self.graph[task] = []

    def get_layers(self):
        """Returns nodes layer by layer (topological levels)"""
        queue = deque()
        layers = []

        # Find all nodes with zero in-degree (independent tasks)
        for node in self.graph:
            if self.in_degree[node] == 0:
                queue.append(node)

        while queue:
            layer = list(queue)  # Current layer
            layers.append(layer)

            for _ in range(len(queue)):
                node = queue.popleft()
                for neighbor in self.graph[node]:
                    self.in_degree[neighbor] -= 1
                    if self.in_degree[neighbor] == 0:
                        queue.append(neighbor)

        return layers