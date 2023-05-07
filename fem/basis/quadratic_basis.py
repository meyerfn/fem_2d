import numpy as np

from .basis import BasisFunctions


class QuadraticBasisFunctions(BasisFunctions):
    def number_of_basis_functions(self) -> int:
        return 6

    def local_basis_functions(self, x, y) -> np.array:
        return np.array(
            [
                [(1 - x - y) * (1 - 2 * x - 2 * y)],
                [x * (2 * x - 1)],
                [y * (2 * y - 1)],
                [4 * x * y],
                [4 * y * (1 - x - y)],
                [4 * x * (1 - x - y)],
            ]
        )

    def local_basis_functions_gradient(self, x, y) -> np.array:
        return np.array(
            [
                [
                    -3 + 4 * x + 4 * y,
                    4 * x - 1,
                    0.0,
                    4 * y,
                    -4 * y,
                    4 - 8 * x - 4 * y,
                ],
                [
                    -3 + 4 * x + 4 * y,
                    0.0,
                    4 * y - 1,
                    4 * x,
                    4 - 4 * x - 8 * y,
                    -4 * x,
                ],
            ]
        )
