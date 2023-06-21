import itertools

import numpy as np
import pytest

from fem.basis import LinearBasisFunctions
from fem.error import Gauss4x4Quadrature, compute_L2_error, compute_L2_error_quadrature
from fem.mesh import LinearMesh
from fem.simulator import Simulator


def set_up_nodes(number_of_nodes_1d: int) -> None:
    x = np.linspace(0, 1, number_of_nodes_1d, endpoint=True)
    nodes = []
    for r in itertools.chain(itertools.product(x, x)):
        nodes.append(r)
    return np.asarray(nodes)


def dirichlet_data(x, y):
    return 1 + x**2 + 2.0 * y**2


def neumann_data(x, y):
    return 0.0


def exact_solution(x, y):
    return 1 + x**2 + 2.0 * y**2


def rhs(x, y):
    return np.ones_like(x) * -6.0


@pytest.mark.parametrize(
    ["quadrature_rule", "expected_l2_errors"],
    [
        (
            None,
            np.array(
                [
                    0.5270462766947298,
                    0.057188368133555115,
                    0.010295844087722696,
                    0.0021949913941173185,
                    0.0005155384834049227,
                ]
            ),
        ),
        (
            Gauss4x4Quadrature(),
            np.array(
                [
                    0.5270462766486744,
                    0.057188368155245604,
                    0.010295844122798898,
                    0.0021949914311107733,
                    0.0005155385220791442,
                ]
            ),
        ),
    ],
)
def test_integration_same_l2_error_linear_elements(quadrature_rule, expected_l2_errors):
    l2_errors = []
    for number_of_nodes_1d in [2, 4, 8, 16, 32]:
        nodes = set_up_nodes(number_of_nodes_1d)
        neumann_edges = []
        simulator = Simulator(
            mesh=LinearMesh(nodes, neumann_edges),
            basis_functions=LinearBasisFunctions(),
            dirichlet_data=dirichlet_data,
            neumann_data=neumann_data,
            rhs=rhs,
            quadrature_rule=quadrature_rule,
        )
        simulator.simulate()
        l2_errors.append(
            compute_L2_error(
                mesh=simulator.mesh,
                coefficients=simulator.solution,
                basis_functions=LinearBasisFunctions(),
                exact_solution=exact_solution,
                quadrature_rule=quadrature_rule,
            )
        )

    np.testing.assert_allclose(expected_l2_errors, l2_errors, rtol=1e-12, atol=1e-15)


# def test_integration_same_l2_error_linear_elements_with_full_quadrature():
#     expected_l2_errors = np.array(
#         [
#             0.5270462766486744,
#             0.057188368155245604,
#             0.010295844122798898,
#             0.0021949914311107733,
#             0.0005155385220791442,
#         ]
#     )
#     l2_errors = []
#     for number_of_nodes_1d in [2, 4, 8, 16, 32]:
#         nodes = set_up_nodes(number_of_nodes_1d)
#         neumann_edges = []
#         simulator = Simulator(
#             mesh=LinearMesh(nodes, neumann_edges),
#             basis_functions=LinearBasisFunctions(),
#             dirichlet_data=dirichlet_data,
#             neumann_data=neumann_data,
#             rhs=rhs,
#             quadrature_rule=Gauss4x4Quadrature(),
#         )
#         simulator.simulate()
#         l2_errors.append(
#             compute_L2_error(
#                 mesh=simulator.mesh,
#                 coefficients=simulator.solution,
#                 basis_functions=LinearBasisFunctions(),
#                 exact_solution=exact_solution,
#                 quadrature_rule=Gauss4x4Quadrature(),
#             )
#         )
#     np.testing.assert_allclose(expected_l2_errors, l2_errors, rtol=1e-14, atol=1e-15)
