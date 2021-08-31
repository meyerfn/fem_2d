from typing import Callable
from mesh.mesh import Mesh
import numpy as np
from assembly.stiffness_matrix import (
    compute_stiffnessmatrix,
)
from assembly.load_vector import (
    compute_loadvector,
)
from basis.basis import BasisFunctions


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
        self.solution = np.zeros(shape=(self.mesh.number_of_nodes, 1))

    def simulate(self):
        A = compute_stiffnessmatrix(self.mesh, self.basis_functions)
        f = compute_loadvector(
            self.rhs,
            self.dirichlet_data,
            self.neumann_data,
            self.mesh,
        )
        self.solution = np.linalg.solve(A, f)
