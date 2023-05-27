import logging
from itertools import product

import numpy as np
import scipy.integrate as integrate

from fem.basis.basis import BasisFunctions
from fem.mesh.mesh import Mesh

logger = logging.getLogger()


def compute_stiffnessmatrix(mesh: Mesh, basis_functions: BasisFunctions) -> np.array:
    logger.info("Compute stiffness maxtrix")
    stiffness_matrix = np.zeros(
        shape=(
            mesh.number_of_nodes,
            mesh.number_of_nodes,
        )
    )
    for index in range(mesh.number_of_elements):
        number_of_basis_functions = basis_functions.number_of_basis_functions()
        local_stiffness_matrix = np.zeros(shape=(number_of_basis_functions, number_of_basis_functions))
        inv_transposed_jacobian = np.linalg.inv(np.transpose(mesh.jacobian[index]))
        for alpha, beta in product(
            range(number_of_basis_functions),
            range(number_of_basis_functions),
        ):

            def integrand(x, y):
                return (
                    np.dot(
                        np.matmul(
                            inv_transposed_jacobian,
                            basis_functions.local_basis_functions_gradient(x, y)[:, alpha],
                        ),
                        np.matmul(
                            inv_transposed_jacobian,
                            basis_functions.local_basis_functions_gradient(x, y)[:, beta],
                        ),
                    )
                    * mesh.determinant[index]
                )

            local_stiffness_matrix[alpha, beta] = integrate.dblquad(
                integrand, a=0, b=1, gfun=lambda x: 0, hfun=lambda x: 1 - x
            )[0]
        local_to_global = mesh.connectivitymatrix[index]
        stiffness_matrix[np.ix_(local_to_global, local_to_global)] += local_stiffness_matrix
    stiffness_matrix = remove_dirichlet_nodes(stiffness_matrix, mesh.boundary_indices)
    return stiffness_matrix


def remove_dirichlet_nodes(
    stiffness_matrix: np.array,
    boundary_indices: list,
) -> np.array:
    for row in boundary_indices:
        stiffness_matrix[row, :] = np.eye(1, len(stiffness_matrix), row)
    return stiffness_matrix
