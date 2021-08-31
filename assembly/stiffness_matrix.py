from mesh.mesh import Mesh
from basis.basis import BasisFunctions
import numpy as np
from itertools import product
import quadpy


def compute_stiffnessmatrix(mesh: Mesh, basis_functions: BasisFunctions) -> np.array:
    stiffness_matrix = np.zeros(
        shape=(
            mesh.number_of_nodes,
            mesh.number_of_nodes,
        )
    )
    for i in range(mesh.number_of_elements):
        local_stiffness_matrix = compute_local_stiffnessmatrix(mesh, basis_functions, i)
        local_to_global = mesh.connectivitymatrix[i]
        stiffness_matrix[np.ix_(local_to_global, local_to_global)] += local_stiffness_matrix
    stiffness_matrix = remove_dirichlet_nodes(stiffness_matrix, mesh.boundary_indices)
    return stiffness_matrix


def compute_local_stiffnessmatrix(mesh: Mesh, basis_functions: BasisFunctions, index: int) -> np.array:
    scheme = quadpy.t2.get_good_scheme(1)
    local_stiffnessmatrix = np.zeros(
        shape=(basis_functions.number_of_basis_functions(), basis_functions.number_of_basis_functions())
    )
    invTransposedJacobian = np.linalg.inv(np.transpose(mesh.jacobian[index]))
    for alpha, beta in product(
        range(basis_functions.number_of_basis_functions()), range(basis_functions.number_of_basis_functions())
    ):

        def integrand(xi):
            return [
                np.dot(
                    np.matmul(
                        invTransposedJacobian,
                        basis_functions.local_basis_functions_gradient(xi)[:, alpha],
                    ),
                    np.matmul(
                        invTransposedJacobian,
                        basis_functions.local_basis_functions_gradient(xi)[:, beta],
                    ),
                )
            ]

        local_stiffnessmatrix[alpha, beta] = scheme.integrate(
            integrand, mesh.nodes[mesh.connectivitymatrix[index, :]]
        )
    return local_stiffnessmatrix


def remove_dirichlet_nodes(
    stiffness_matrix: np.array,
    boundary_indices: list,
) -> np.array:
    for row in boundary_indices:
        stiffness_matrix[row, :] = np.eye(1, len(stiffness_matrix), row)
    return stiffness_matrix
