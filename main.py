from mesh.mesh import Mesh
from simulator.simulator import Simulator
from basis.linear_basis import LinearBasisFunctions
from plot.plot_solution import plot_solution
from analyze.error import compute_L2_error
import itertools
import numpy as np


def set_up_nodes(number_of_nodes_1d: int) -> None:
    x = np.linspace(0, 1, number_of_nodes_1d, endpoint=True)
    nodes = []
    for r in itertools.chain(itertools.product(x, x)):
        nodes.append(r)
    return np.asarray(nodes)


# def rhs(node):
#     return 1.0

# def dirichlet_data(node):
#     return 0.0


def rhs(node):
    return -6.0


def dirichlet_data(node):
    return 1 + node[0] ** 2 + 2.0 * node[1] ** 2


def neumann_data(node):
    return 0.0


def exact_solution(node):
    return 1 + node[0] ** 2 + 2.0 * node[1] ** 2


if __name__ == "__main__":
    nodes = set_up_nodes(32)
    neumann_edges = np.array([[[0, 0], [0, 1]], [[0, 0], [1, 0]]])
    # neumann_edges = []
    simulator = Simulator(
        Mesh(nodes, neumann_edges), LinearBasisFunctions(), dirichlet_data, neumann_data, rhs
    )
    simulator.simulate()
    plot_solution(simulator)
    l2_error = compute_L2_error(simulator.mesh, simulator.solution, LinearBasisFunctions(), exact_solution)
    np.savetxt(f'{"error.txt"}', X=[l2_error])
