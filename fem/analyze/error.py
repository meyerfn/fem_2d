import numpy as np
import quadpy

from fem.basis.basis import BasisFunctions
from fem.mesh.mesh import Mesh


def compute_L2_error(mesh: Mesh, coefficients: np.array, basis_functions: BasisFunctions, exact_solution=0.0):
    scheme = quadpy.t2.get_good_scheme(1)
    l2_error = 0.0
    for index in range(mesh.number_of_elements):

        def integrand(xi):
            [p1, p2, p3] = mesh.nodes[mesh.connectivitymatrix[index, :]][0:3]
            transformed_points = np.add(p1, np.matmul(np.array([p2 - p1, p3 - p1]), xi).flatten())
            local_coefficients = coefficients[mesh.connectivitymatrix[index, :]]
            local_approximation = np.sum(
                [
                    local_coefficients[i] * basis_functions.local_basis_functions(xi)[i]
                    for i in range(basis_functions.number_of_basis_functions())
                ]
            )
            return [mesh.determinant[index] * (exact_solution(transformed_points) - local_approximation) ** 2]

        l2_error += scheme.integrate(integrand, np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]]))
    return np.sqrt(l2_error)
