from typing import Callable

import numpy as np

from fem.basis import BasisFunctions
from fem.load_vector import compute_loadvector, compute_loadvector_int
from fem.mesh import Mesh
from fem.stiffness_matrix import compute_stiffnessmatrix


class Simulator:
    def __init__(
        self,
        mesh: Mesh,
        basis_functions: BasisFunctions,
        dirichlet_data: Callable,
        neumann_data: Callable,
        rhs: Callable,
    ) -> None:
        self.mesh = mesh
        self.basis_functions = basis_functions
        self.dirichlet_data = dirichlet_data
        self.neumann_data = neumann_data
        self.rhs = rhs
        self.solution = np.zeros(shape=(self.mesh.number_of_nodes))

    def simulate(self):
        A = compute_stiffnessmatrix(self.mesh, self.basis_functions)
        f = compute_loadvector_int(
            self.rhs,
            self.basis_functions,
            self.dirichlet_data,
            self.neumann_data,
            self.mesh,
        )
        self.solution = np.linalg.solve(A, f)
