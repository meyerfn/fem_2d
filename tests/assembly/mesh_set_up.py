import unittest
from fem.mesh.mesh import Mesh
import numpy as np
import itertools


class MeshSetUp(unittest.TestCase):
    def setUp(self) -> None:
        number_of_nodes_1d = 3
        self.setup_nodes_and_neumann_edges(number_of_nodes_1d)
        self.mesh = Mesh(self.nodes, self.neumann_edges)

    def setup_nodes_and_neumann_edges(self, number_of_nodes_1d: int) -> None:
        x = np.linspace(0, 1, number_of_nodes_1d, endpoint=True)
        self.nodes = []
        for r in itertools.chain(itertools.product(x, x)):
            self.nodes.append(r)
        self.nodes = np.asarray(self.nodes)
        self.neumann_edges = np.array([[[0, 0], [0, 1]], [[0, 0], [1, 0]]])
