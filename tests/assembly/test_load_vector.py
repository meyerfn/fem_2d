import unittest
from mesh_set_up import MeshSetUp
from fem.assembly.load_vector import (
    compute_loadvector,
)


class LoadVectorUnittest(MeshSetUp):
    def test_computation_of_load_vector(self):
        def rhs(node):
            return -6.0

        def dirichlet_data(node):
            return 1.0 + node[0] ** 2 + 2.0 * node[1] ** 2

        def neumann_data(node):
            return 0.0

        load_vector = compute_loadvector(rhs, dirichlet_data, neumann_data, self.mesh)
        self.assertEqual(
            load_vector[self.mesh.free_indices],
            4.0 * 1.0 / 3.0 * 0.5 * self.mesh.determinant[0] * rhs(self.mesh.pointmatrix[4]),
        )


if __name__ == "__main__":
    unittest.main()
