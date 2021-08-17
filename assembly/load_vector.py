from mesh.mesh import Mesh
from typing import Callable

import numpy as np


def compute_loadvector(
    rhs: Callable,
    dirichlet_data: Callable,
    neumann_data: Callable,
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
    load_vector = add_dirichlet_data(load_vector, dirichlet_data, mesh)
    load_vector = add_neumann_data(load_vector, neumann_data, mesh)
    return load_vector


def add_dirichlet_data(
    load_vector: np.array,
    dirichlet_data: Callable,
    mesh: Mesh,
) -> np.array:
    for index in mesh.boundary_indices:
        load_vector[index] = dirichlet_data(mesh.nodes[index])
    return load_vector


def add_neumann_data(
    load_vector: np.array,
    neumann_data: Callable,
    mesh: Mesh,
) -> np.array:
    for neumann_edge_idx in mesh.neumann_edges:
        length_of_edge = np.linalg.norm(mesh.nodes[neumann_edge_idx[1]] - mesh.nodes[neumann_edge_idx[0]], 2)
        mid_point = 0.5 * (mesh.nodes[neumann_edge_idx[0]] + mesh.nodes[neumann_edge_idx[1]])
        load_vector[neumann_edge_idx] += 0.5 * neumann_data(mid_point) * length_of_edge
    return load_vector
