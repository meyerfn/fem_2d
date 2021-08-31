from basis.basis import BasisFunctions
import numpy as np


class LinearBasisFunctions(BasisFunctions):
    def number_of_basis_functions(self) -> int:
        return 3

    def local_basis_functions(self, xi) -> np.array:
        return np.array([[1 - xi[0] - xi[1]], [xi[0]], [xi[1]]])

    def local_basis_functions_gradient(self, xi) -> np.array:
        return np.array([[-1.0, 1.0, 0.0], [-1.0, 0.0, 1.0]])
