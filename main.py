from mesh.mesh import Mesh
from simulator.simulator import Simulator
from plot.plot_solution import plot_solution

import itertools
import numpy as np


def setUpNodes(number_of_nodes_1d: int) -> None:
    x = np.linspace(0, 1, number_of_nodes_1d, endpoint=True)
    nodes = []
    for r in itertools.chain(itertools.product(x, x)):
        nodes.append(r)
    return np.asarray(nodes)


def rhs(node):
    return -6.0


# def rhs(node):
#     return 1.0


def dirichlet_data(node):
    return 1 + node[0] ** 2 + 2.0 * node[1] ** 2


# def dirichlet_data(node):
#     return 0.0


def neumann_data(node):
    return 1.0


if __name__ == "__main__":
    x = np.linspace(0, 1, 10, endpoint=True)
    nodes = []
    for r in itertools.chain(itertools.product(x, x)):
        nodes.append(r)
    nodes = np.asarray(nodes)
    neumann_edges = np.array([[[0, 0], [0, 1]], [[0, 0], [1, 0]]])
    # neumann_edges = []
    mesh = Mesh(nodes, neumann_edges)
    simulator = Simulator(mesh, dirichlet_data, neumann_data, rhs)
    simulator.simulate()
    plot_solution(simulator)
