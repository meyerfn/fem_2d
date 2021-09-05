from typing import Callable

import numpy as np
import quadpy

from fem.basis.basis import BasisFunctions
from fem.mesh.mesh import Mesh


def compute_loadvector_int(
    rhs: Callable,
    basis_functions: BasisFunctions,
    dirichlet_data: Callable,
    neumann_data: Callable,
    mesh: Mesh,
) -> np.array:
    load_vector = np.zeros(shape=(mesh.number_of_nodes, 1))
    number_of_basis_functions = basis_functions.number_of_basis_functions()
    local_load_vector = np.zeros(shape=(number_of_basis_functions, 1))
    scheme = quadpy.t2.get_good_scheme(3)
    for i in range(mesh.number_of_elements):

        def integrand(xi):
            return mesh.determinant[i] * rhs(xi) * basis_functions.local_basis_functions(xi)

        local_load_vector = scheme.integrate(integrand, np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]]))
        local_to_global = mesh.connectivitymatrix[i]
        load_vector[np.ix_(local_to_global)] += local_load_vector
    load_vector = add_dirichlet_data(load_vector, dirichlet_data, mesh)
    load_vector = add_neumann_data(load_vector, neumann_data, mesh)
    return load_vector


def compute_loadvector(
    rhs: Callable,
    dirichlet_data: Callable,
    neumann_data: Callable,
    mesh: Mesh,
) -> np.array:
    load_vector = np.zeros(shape=(mesh.number_of_nodes, 1))
    for i in range(mesh.number_of_elements):
        (node_one, node_two, node_three, _) = mesh.pointmatrix[i]
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
