import itertools
import logging
import time

import numpy as np
import scipy.integrate as integrate
import scipy.sparse

from fem.basis import BasisFunctions
from fem.mesh import Mesh

logger = logging.getLogger()


def compute_stiffnessmatrix(mesh: Mesh, basis_functions: BasisFunctions) -> np.array:
    start_time = time.perf_counter()
    logger.info("Compute stiffness maxtrix")
    number_of_basis_functions = basis_functions.number_of_basis_functions()
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
    inv_jacobian = np.linalg.inv(mesh.jacobian)
    scaling_matrix = inv_jacobian @ np.transpose(inv_jacobian, axes=(0, 2, 1))
    scaling_matrix = scaling_matrix.reshape((scaling_matrix.shape[0], -1))
    stiffness_matrix = np.array([K_xx.ravel(), K_xy.ravel(), K_xy.T.ravel(), K_yy.ravel()]).T
    result = mesh.determinant * (stiffness_matrix @ scaling_matrix.T)
    row = np.repeat(mesh.connectivitymatrix, repeats=3)
    col = np.repeat(mesh.connectivitymatrix, repeats=3, axis=0).ravel()
    stiff_matrix = scipy.sparse.coo_matrix(
        (result.ravel("F"), (row, col)),
        shape=(
            mesh.number_of_nodes,
            mesh.number_of_nodes,
        ),
    )
    stiff_matrix = stiff_matrix.tocsr()
    stiff_matrix = remove_dirichlet_nodes(stiff_matrix, mesh.boundary_indices)
    end_time = time.perf_counter()
    logger.info(f"Computation took {end_time-start_time} seconds")
    return stiff_matrix


def remove_dirichlet_nodes(
    stiffness_matrix: np.array,
    boundary_indices: list,
) -> np.array:
    for row in boundary_indices:
        stiffness_matrix[row, :] = np.eye(1, stiffness_matrix.shape[0], row)
    return stiffness_matrix
