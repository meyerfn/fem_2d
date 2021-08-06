from mesh.mesh import Mesh
from typing import Callable

import numpy as np


def compute_loadvector(
    rhs: Callable,
    dirichlet_data: Callable,
    mesh: Mesh,
) -> np.array:
    load_vector = np.zeros(shape=(mesh.number_of_nodes, 1))
    for i, _ in enumerate(mesh.pointmatrix, start=0):
        (
            node_one,
            node_two,
            node_three,
        ) = mesh.pointmatrix[i]
        area = 0.5 * mesh.determinant[i]
        local_load_vector = (
            1.0
            / 3.0
            * area
            * np.array(
                [
                    [rhs(node_one)],
                    [rhs(node_two)],
                    [rhs(node_three)],
                ]
            )
        )
        local_to_global = mesh.connectivitymatrix[i]
        load_vector[np.ix_(local_to_global)] += local_load_vector
    load_vector = adapt_dirichlet_data(load_vector, dirichlet_data, mesh)
    return load_vector


def adapt_dirichlet_data(
    load_vector: np.array,
    dirichlet_data: Callable,
    mesh: Mesh,
) -> np.array:
    for index in mesh.boundary_indices:
        load_vector[index] = dirichlet_data(mesh.nodes[index])
    return load_vector
