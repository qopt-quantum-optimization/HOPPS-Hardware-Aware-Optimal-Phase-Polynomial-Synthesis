'''Copyright © 2025 UChicago Argonne, LLC and Case Western Reserve University All right reserved

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at:

    https://github.com/qopt-quantum-optimization/HOPPS-Hardware-Aware-Optimal-Phase-Polynomial-Synthesis/blob/main/LICENSE.md

Unless required by applicable law or
agreed to in writing, Licensor provides the Work (and each
Contributor provides its Contributions) on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
implied, including, without limitation, any warranties or conditions
of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
PARTICULAR PURPOSE. You are solely responsible for determining the
appropriateness of using or redistributing the Work and assume any
risks associated with Your exercise of permissions under this License.'''

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