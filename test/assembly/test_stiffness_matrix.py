import unittest
import numpy as np
from mesh_set_up import MeshSetUp
from assembly.stiffness_matrix import (
    compute_stiffnessmatrix,
)


class MatrixAssemblyUnittest(MeshSetUp):
    def test_assemble_matrix(self):
        stiffness_matrix = compute_stiffnessmatrix(self.mesh)
        np.testing.assert_array_almost_equal(
            stiffness_matrix[self.mesh.free_indices[0], :],
            np.array([0, -1, 0, -1, 4, -1, 0, -1, 0]),
        )


if __name__ == "__main__":
    unittest.main()
