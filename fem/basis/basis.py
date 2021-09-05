from abc import ABC, abstractclassmethod

import numpy as np


class BasisFunctions(ABC):
    @abstractclassmethod
    def number_of_basis_functions() -> int:
        pass

    @abstractclassmethod
    def local_basis_functions(self, xi: np.array) -> np.array:
        pass

    @abstractclassmethod
    def local_basis_functions_gradient(self, xi) -> np.array:
        pass
