import logging
import time
from abc import ABC, abstractmethod

import numpy as np
import scipy.integrate as integrate

from fem.basis import BasisFunctions
from fem.mesh import Mesh
from fem.utils import timing

logger = logging.getLogger()


class QuadratureRule(ABC):
    @property
    @abstractmethod
    def points(self) -> np.array:
        pass

    @property
    @abstractmethod
    def weights(self) -> np.array:
        pass


class CentroidQuadrature(QuadratureRule):
    @property
    def points(self) -> np.array:
        return np.array([[0.33333333333333333333, 0.33333333333333333333]])

    @property
    def weights(self) -> np.array:
        return 0.5 * np.array([1.0])


class Gauss4x4Quadrature(QuadratureRule):
    @property
    def points(self) -> np.array:
        return np.array(
            [
                [0.0571041961, 0.06546699455602246],
                [0.2768430136, 0.05021012321401679],
                [0.5835904324, 0.02891208422223085],
                [0.8602401357, 0.009703785123906346],
                [0.0571041961, 0.3111645522491480],
                [0.2768430136, 0.2386486597440242],
                [0.5835904324, 0.1374191041243166],
                [0.8602401357, 0.04612207989200404],
                [0.0571041961, 0.6317312516508520],
                [0.2768430136, 0.4845083266559759],
                [0.5835904324, 0.2789904634756834],
                [0.8602401357, 0.09363778440799593],
                [0.0571041961, 0.8774288093439775],
                [0.2768430136, 0.6729468631859832],
                [0.5835904324, 0.3874974833777692],
                [0.8602401357, 0.1300560791760936],
            ]
        )

    @property
    def weights(self) -> np.array:
        return 0.5 * np.array(
            [
                0.04713673637581137,
                0.07077613579259895,
                0.04516809856187617,
                0.01084645180365496,
                0.08837017702418863,
                0.1326884322074010,
                0.08467944903812383,
                0.02033451909634504,
                0.08837017702418863,
                0.1326884322074010,
                0.08467944903812383,
                0.02033451909634504,
                0.04713673637581137,
                0.07077613579259895,
                0.04516809856187617,
                0.01084645180365496,
            ]
        )


def compute_L2_error(
    mesh: Mesh,
    coefficients: np.array,
    basis_functions: BasisFunctions,
    exact_solution=0.0,
    quadrature_rule: QuadratureRule = None,
) -> float:
    if quadrature_rule:
        return compute_L2_error_quadrature(
            mesh=mesh,
            coefficients=coefficients,
            basis_functions=basis_functions,
            exact_solution=exact_solution,
            quadrature_rule=quadrature_rule,
        )
    else:
        return compute_L2_error_scipy(
            mesh=mesh,
            coefficients=coefficients,
            basis_functions=basis_functions,
            exact_solution=exact_solution,
        )


@timing
def compute_L2_error_scipy(
    mesh: Mesh, coefficients: np.array, basis_functions: BasisFunctions, exact_solution=0.0
) -> float:
    logger.info("Compute L2-error")
    l2_error = 0.0
    for index in range(mesh.number_of_elements):
        [p1, p2, p3] = mesh.nodes[mesh.connectivitymatrix[index, :]][0:3]
        local_coefficients = coefficients[mesh.connectivitymatrix[index, :]]

        def integrand(x, y):
            transformed_points = np.add(
                p1, np.matmul(np.array([p2 - p1, p3 - p1]).T, np.array([x, y])).flatten()
            )
            local_approximation = np.dot(local_coefficients, basis_functions.local_basis_functions(x, y))
            return (exact_solution(*transformed_points) - local_approximation) ** 2

        l2_error += (
            mesh.determinant[index]
            * integrate.dblquad(integrand, a=0, b=1, gfun=lambda x: 0, hfun=lambda x: 1 - x)[0]
        )
    return np.sqrt(l2_error)


@timing
def compute_L2_error_quadrature(
    mesh: Mesh,
    coefficients: np.array,
    basis_functions: BasisFunctions,
    quadrature_rule: QuadratureRule,
    exact_solution=0.0,
) -> float:
    logger.info("Compute L2-error")
    l2_error = 0.0
    quadrature_points = quadrature_rule.points
    quadrature_matrix = np.array(
        [
            basis_functions.local_basis_functions(*quadrature_point).flatten()
            for quadrature_point in quadrature_points
        ]
    )
    p1_vec = mesh.pointmatrix[:, 0, :]
    transformed_points = np.transpose(
        p1_vec[..., None] + np.matmul(mesh.transformationmatrix, quadrature_rule.points.T), axes=(0, 2, 1)
    )
    exact_solution_quadrature_points = exact_solution(
        transformed_points[:, :, 0], transformed_points[:, :, 1]
    )
    local_approximation = np.dot(coefficients[mesh.connectivitymatrix], quadrature_matrix.T)
    local_error = np.dot(
        (local_approximation - exact_solution_quadrature_points) ** 2, quadrature_rule.weights
    )
    l2_error = np.sum(mesh.determinant * local_error)
    return np.sqrt(l2_error)
