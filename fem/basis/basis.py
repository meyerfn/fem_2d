from abc import ABC, abstractclassmethod

import numpy as np


class BasisFunctions(ABC):
    @abstractclassmethod
    def number_of_basis_functions() -> int:
        pass

    @abstractclassmethod
    def local_basis_functions(self, x, y: np.array) -> np.array:
        pass

    @abstractclassmethod
    def local_basis_functions_gradient(self, x, y) -> np.array:
        pass
