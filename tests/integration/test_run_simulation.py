import itertools

import numpy as np

from fem.error import compute_L2_error
from fem.basis import LinearBasisFunctions
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
    return -6.0


def test_integration_same_l2_error_linear_elements():
    expected_l2_errors = np.array(
        [
            1.173787790777267359e00,
            4.601491134399846028e-01,
            1.986708535724402147e-01,
            8.991444503855471060e-02,
            4.366051680121468825e-02,
        ]
    )
    l2_errors = []
    for number_of_nodes_1d in [2, 4, 8, 16, 32]:
        nodes = set_up_nodes(number_of_nodes_1d)
        neumann_edges = np.array([[[0, 0], [0, 1]], [[0, 0], [1, 0]]])
        neumann_edges = []
        simulator = Simulator(
            LinearMesh(nodes, neumann_edges), LinearBasisFunctions(), dirichlet_data, neumann_data, rhs
        )
        simulator.simulate()
        l2_errors.append(
            compute_L2_error(simulator.mesh, simulator.solution, LinearBasisFunctions(), exact_solution)
        )
    np.testing.assert_allclose(expected_l2_errors, l2_errors, rtol=1e-14, atol=1e-15)
