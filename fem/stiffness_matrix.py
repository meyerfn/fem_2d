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
    result = (mesh.determinant * (stiffness_matrix @ scaling_matrix.T)).ravel("F")
    row = np.repeat(mesh.connectivitymatrix, repeats=3)
    col = np.repeat(mesh.connectivitymatrix, repeats=3, axis=0).ravel()
    stiff_matrix = scipy.sparse.coo_matrix(
        (result, (row, col)),
    )
    stiff_matrix = stiff_matrix.tocsr()
    boundary_indices = np.array(list(mesh.boundary_indices), dtype=int)
    stiff_matrix = remove_dirichlet_nodes(stiff_matrix, boundary_indices)
    end_time = time.perf_counter()
    logger.info(f"Computation took {end_time-start_time} seconds")
    return stiff_matrix


def remove_dirichlet_nodes(A: scipy.sparse.csr_matrix, bc_id: np.array) -> scipy.sparse.csr_matrix:
    ndofs = A.shape[0]
    eye_like_matrix = np.ones((ndofs))
    eye_like_matrix[bc_id] = 0
    eye_like_matrix = scipy.sparse.dia_matrix((eye_like_matrix, 0), shape=(ndofs, ndofs))
    # up to here I delete the rows
    # I multiply A by an identity matrix
    # where i set to zero the rows I want
    # to delete
    A = eye_like_matrix.dot(A)
    new_diag_entries = np.zeros((ndofs))
    new_diag_entries[bc_id] = 1.0
    eye_like_matrix = scipy.sparse.dia_matrix((new_diag_entries, 0), shape=(ndofs, ndofs))
    A = A + eye_like_matrix  # here I set the diagonal entry
    return A
