import itertools
import logging
from pathlib import Path

import numpy as np

from fem.basis import LinearBasisFunctions, QuadraticBasisFunctions
from fem.error import CentroidQuadrature, Gauss4x4Quadrature, compute_L2_error
from fem.mesh import LinearMesh, QuadraticMesh
from fem.plot.plot_solution import plot_solution
from fem.simulator import Simulator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

import argparse


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


def rhs(x, y):
    return np.ones_like(x) * -6.0


def dirichlet_data(x, y):
    return 1 + x**2 + 2.0 * y**2


def neumann_data(x, y):
    return 0.0


def exact_solution(x, y):
    return 1 + x**2 + 2.0 * y**2


def main():
    paser = argparse.ArgumentParser()
    paser.add_argument("-o", "--output_directory", dest="output_directory", default=None)
    args = paser.parse_args()
    for number_of_nodes_1d in [2, 4, 8, 16, 32, 64]:
        logger.info(f"Compute solution for number_of_nodes_1d {number_of_nodes_1d}")
        nodes = set_up_nodes(number_of_nodes_1d)
        neumann_edges = np.array([[[0, 0], [0, 1]], [[0, 0], [1, 0]]])
        neumann_edges = []
        simulator = Simulator(
            mesh=LinearMesh(nodes, neumann_edges),
            basis_functions=LinearBasisFunctions(),
            dirichlet_data=dirichlet_data,
            neumann_data=neumann_data,
            rhs=rhs,
            quadrature_rule=Gauss4x4Quadrature(),
        )
        simulator.simulate()
        # plot_solution(simulator)
        l2_error = compute_L2_error(
            mesh=simulator.mesh,
            coefficients=simulator.solution,
            basis_functions=LinearBasisFunctions(),
            exact_solution=exact_solution,
            quadrature_rule=Gauss4x4Quadrature(),
        )
        if args.output_directory:
            logger.info(f"Save result to {args.output_directory}")
            np.savetxt(Path(args.output_directory) / f"error_{number_of_nodes_1d}.txt", X=[l2_error])


if __name__ == "__main__":
    main()
