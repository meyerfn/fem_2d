import logging

import numpy as np
import scipy.integrate as integrate

from fem.basis.basis import BasisFunctions
from fem.mesh.mesh import Mesh

logger = logging.getLogger()


def compute_L2_error(mesh: Mesh, coefficients: np.array, basis_functions: BasisFunctions, exact_solution=0.0):
    logger.info("Compute L2-error")
    l2_error = 0.0
    for index in range(mesh.number_of_elements):

        def integrand(x, y):
            # edge points of triangle
            [p1, p2, p3] = mesh.nodes[mesh.connectivitymatrix[index, :]][0:3]
            transformed_points = np.add(
                p1, np.matmul(np.array([p2 - p1, p3 - p1]), np.array([x, y])).flatten()
            )
            local_coefficients = coefficients[mesh.connectivitymatrix[index, :]]
            local_approximation = np.dot(local_coefficients, basis_functions.local_basis_functions(x, y))
            return mesh.determinant[index] * (exact_solution(*transformed_points) - local_approximation) ** 2

        l2_error += integrate.dblquad(integrand, a=0, b=1, gfun=lambda x: 0, hfun=lambda x: 1 - x)[0]
    return np.sqrt(l2_error)
