import numpy as np

from .basis import BasisFunctions


class QuadraticBasisFunctions(BasisFunctions):
    def number_of_basis_functions(self) -> int:
        return 6

    def local_basis_functions(self, xi) -> np.array:
        return np.array(
            [
                [(1 - xi[0] - xi[1]) * (1 - 2 * xi[0] - 2 * xi[1])],
                [xi[0] * (2 * xi[0] - 1)],
                [xi[1] * (2 * xi[1] - 1)],
                [4 * xi[0] * xi[1]],
                [4 * xi[1] * (1 - xi[0] - xi[1])],
                [4 * xi[0] * (1 - xi[0] - xi[1])],
            ]
        )

    def local_basis_functions_gradient(self, xi) -> np.array:
        return np.array(
            [
                [
                    -3 + 4 * xi[0] + 4 * xi[1],
                    4 * xi[0] - 1,
                    0.0,
                    4 * xi[1],
                    -4 * xi[1],
                    4 - 8 * xi[0] - 4 * xi[1],
                ],
                [
                    -3 + 4 * xi[0] + 4 * xi[1],
                    0.0,
                    4 * xi[1] - 1,
                    4 * xi[0],
                    4 - 4 * xi[0] - 8 * xi[1],
                    -4 * xi[0],
                ],
            ]
        )
