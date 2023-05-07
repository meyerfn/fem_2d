import itertools
import unittest

import matplotlib.pyplot as plt
import numpy as np

import fem.mesh.mesh as mesh
from fem.mesh.mesh import QuadraticMesh


class UnittestMesh(unittest.TestCase):
    def setUp(self) -> None:
        number_of_nodes_1d = 3
        self.setup_nodes_and_neumann_edges(number_of_nodes_1d)
        self.mesh = mesh.Mesh(self.nodes, self.neumann_edges)

    def setup_nodes_and_neumann_edges(self, number_of_nodes_1d: int) -> None:
        x = np.linspace(0, 1, number_of_nodes_1d, endpoint=True)
        self.nodes = []
        for r in itertools.chain(itertools.product(x, x)):
            self.nodes.append(r)
        self.nodes = np.asarray(self.nodes)
        self.neumann_edges = np.array([[[0, 0], [0, 1]], [[0, 0], [1, 0]]])

    def test_mesh_is_correctly_set_up(self):
        self.assertEqual(self.mesh.number_of_nodes, 9)
        self.assertEqual(self.mesh.number_of_elements, 8)
        self.assertEqual(
            self.mesh.determinant,
            (0.25 * np.ones(8)).tolist(),
        )
        expected_boundary_indices = [
            0,
            1,
            2,
            3,
            5,
            6,
            7,
            8,
        ]
        self.assertEqual(
            list(self.mesh.boundary_indices),
            expected_boundary_indices,
        )

    def test_mesh_correct_neumann_edges(self):
        expected_neumann_edges = np.asarray([[0, 3], [3, 6], [0, 1], [1, 2]])
        self.assertTrue(
            np.array_equal(np.sort(self.mesh.neumann_edges.flat), np.sort(expected_neumann_edges.flat))
        )

    @unittest.skip
    def test_plot_mesh(self):
        plt.triplot(
            self.nodes[:, 0],
            self.nodes[:, 1],
            self.mesh.connectivitymatrix,
        )
        plt.plot(
            self.nodes[:, 0],
            self.nodes[:, 1],
            "o",
        )
        plt.show()


if __name__ == "__main__":
    unittest.main()


def test_quadratic_mesh_contains_additional_points_on_edges_of_triangles():
    x = np.linspace(0, 1, 8, endpoint=True)
    nodes = []
    for r in itertools.chain(itertools.product(x, x)):
        nodes.append(r)
    nodes = np.asarray(nodes)
    neumann_edges = []
    mesh = QuadraticMesh(nodes=nodes, neumann_edges=neumann_edges)

    assert len(mesh.connectivitymatrix[0]) == 6
    plt.triplot(
        mesh.nodes[:, 0],
        mesh.nodes[:, 1],
        mesh.connectivitymatrix[:, 0:3],
    )
    plt.plot(
        mesh.nodes[:, 0],
        mesh.nodes[:, 1],
        "o",
    )
    plt.show()
