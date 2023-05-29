import logging
import time
from itertools import product

import numpy as np
import scipy.integrate as integrate

from fem.basis import BasisFunctions
from fem.mesh import Mesh

logger = logging.getLogger()


def compute_stiffnessmatrix(mesh: Mesh, basis_functions: BasisFunctions) -> np.array:
    start_time = time.perf_counter()
    logger.info("Compute stiffness maxtrix")
    stiffness_matrix = np.zeros(
        shape=(
            mesh.number_of_nodes,
            mesh.number_of_nodes,
        )
    )
    number_of_basis_functions = basis_functions.number_of_basis_functions()
    local_stiffness_matrix = np.zeros(shape=(number_of_basis_functions, number_of_basis_functions))
    integrand = (
        lambda y, x, alpha, beta, first_direction, second_direction: basis_functions.local_basis_functions_gradient(
            x, y
        )[
            first_direction, alpha
        ]
        * basis_functions.local_basis_functions_gradient(y, x)[second_direction, beta]
    )
    K_xx = np.array(
        [
            [
                integrate.dblquad(
                    integrand, a=0, b=1, gfun=lambda x: 0, hfun=lambda x: 1 - x, args=(alpha, beta, 0, 0)
                )[0]
                for alpha in range(number_of_basis_functions)
            ]
            for beta in range(number_of_basis_functions)
        ]
    )
    K_yy = np.array(
        [
            [
                integrate.dblquad(
                    integrand, a=0, b=1, gfun=lambda x: 0, hfun=lambda x: 1 - x, args=(alpha, beta, 1, 1)
                )[0]
                for alpha in range(number_of_basis_functions)
            ]
            for beta in range(number_of_basis_functions)
        ]
    )
    K_xy = np.array(
        [
            [
                integrate.dblquad(
                    integrand, a=0, b=1, gfun=lambda x: 0, hfun=lambda x: 1 - x, args=(alpha, beta, 0, 1)
                )[0]
                for alpha in range(number_of_basis_functions)
            ]
            for beta in range(number_of_basis_functions)
        ]
    )
    for index in range(mesh.number_of_elements):
        scaling_matrix = np.linalg.inv((mesh.jacobian[index])) @ np.linalg.inv(
            np.transpose(mesh.jacobian[index])
        )
        local_stiffness_matrix = mesh.determinant[index] * (
            scaling_matrix[0, 0] * K_xx + scaling_matrix[1, 1] * K_yy + scaling_matrix[0, 1] * (K_xy + K_xy.T)
        )
        local_to_global = mesh.connectivitymatrix[index]
        stiffness_matrix[np.ix_(local_to_global, local_to_global)] += local_stiffness_matrix
    stiffness_matrix = remove_dirichlet_nodes(stiffness_matrix, mesh.boundary_indices)
    end_time = time.perf_counter()
    logger.info(f"Computation took {end_time-start_time} seconds")
    return stiffness_matrix


def remove_dirichlet_nodes(
    stiffness_matrix: np.array,
    boundary_indices: list,
) -> np.array:
    for row in boundary_indices:
        stiffness_matrix[row, :] = np.eye(1, len(stiffness_matrix), row)
    return stiffness_matrix
