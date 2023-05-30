from typing import Callable

import numpy as np
import scipy.sparse.linalg

from fem.basis import BasisFunctions
from fem.error import QuadratureRule
from fem.load_vector import compute_loadvector, compute_loadvector_quadrature
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
        quadrature_rule: QuadratureRule = None,
    ) -> None:
        self.mesh = mesh
        self.basis_functions = basis_functions
        self.dirichlet_data = dirichlet_data
        self.neumann_data = neumann_data
        self.rhs = rhs
        self.solution = np.zeros(shape=(self.mesh.number_of_nodes))
        self.quadrature_rule = quadrature_rule

    def simulate(self):
        A = compute_stiffnessmatrix(self.mesh, self.basis_functions)
        f = compute_loadvector(
            rhs=self.rhs,
            basis_functions=self.basis_functions,
            dirichlet_data=self.dirichlet_data,
            neumann_data=self.neumann_data,
            mesh=self.mesh,
            quadrature_rule=self.quadrature_rule,
        )
        self.solution = scipy.sparse.linalg.spsolve(A, f)
