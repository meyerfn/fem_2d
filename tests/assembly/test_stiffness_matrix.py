import unittest

import numpy as np
from mesh_set_up import MeshSetUp

from fem.basis import LinearBasisFunctions
from fem.stiffness_matrix import compute_stiffnessmatrix


class MatrixAssemblyUnittest(MeshSetUp):
    def test_assemble_matrix_linear_basis_functions(self):
        basis_functions = LinearBasisFunctions()
        stiffness_matrix = compute_stiffnessmatrix(self.mesh, basis_functions)
        np.testing.assert_array_almost_equal(
            stiffness_matrix[self.mesh.free_indices[0], :],
            np.array([0, -1, 0, -1, 4, -1, 0, -1, 0]),
        )


if __name__ == "__main__":
    unittest.main()
