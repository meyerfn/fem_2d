import numpy as np

from .basis import BasisFunctions


class LinearBasisFunctions(BasisFunctions):
    def number_of_basis_functions(self) -> int:
        return 3

    def local_basis_functions(self, x, y) -> np.array:
        return np.array(
            [
                [1 - x - y],
                [x],
                [y],
            ]
        )

    def local_basis_functions_gradient(self, x, y) -> np.array:
        return np.array(
            [
                [-1.0, 1.0, 0.0],
                [-1.0, 0.0, 1.0],
            ]
        )
