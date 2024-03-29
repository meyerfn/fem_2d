import logging
import time
from typing import Callable

import numpy as np
import scipy.integrate as integrate
from scipy.sparse import coo_array

from fem.basis import BasisFunctions
from fem.error import QuadratureRule
from fem.mesh import Mesh
from fem.utils import timing

logger = logging.getLogger()


def compute_loadvector(
    rhs: Callable,
    basis_functions: BasisFunctions,
    dirichlet_data: Callable,
    neumann_data: Callable,
    mesh: Mesh,
    quadrature_rule: QuadratureRule = None,
) -> np.array:
    if quadrature_rule:
        return compute_loadvector_quadrature(
            rhs=rhs,
            basis_functions=basis_functions,
            dirichlet_data=dirichlet_data,
            neumann_data=neumann_data,
            mesh=mesh,
            quadrature_rule=quadrature_rule,
        )
    else:
        return compute_loadvector_scipy(
            rhs=rhs,
            basis_functions=basis_functions,
            dirichlet_data=dirichlet_data,
            neumann_data=neumann_data,
            mesh=mesh,
        )


@timing
def compute_loadvector_scipy(
    rhs: Callable,
    basis_functions: BasisFunctions,
    dirichlet_data: Callable,
    neumann_data: Callable,
    mesh: Mesh,
) -> np.array:
    logger.info("Compute load vector")
    load_vector = np.zeros(shape=(mesh.number_of_nodes,))
    number_of_basis_functions = basis_functions.number_of_basis_functions()
    local_load_vector = np.zeros(shape=(number_of_basis_functions,))
    for i in range(mesh.number_of_elements):
        [p1, p2, p3] = mesh.nodes[mesh.connectivitymatrix[i, :]][0:3]

        def integrand(x, y, idx):
            transformed_points = np.add(
                p1, np.matmul(np.array([p2 - p1, p3 - p1]).T, np.array([x, y])).flatten()
            )
            return rhs(*transformed_points) * basis_functions.local_basis_functions(x, y)[idx]

        local_load_vector = mesh.determinant[i] * np.array(
            [
                integrate.dblquad(integrand, a=0, b=1, gfun=lambda x: 0, hfun=lambda x: 1 - x, args=(j,))[0]
                for j in range(basis_functions.number_of_basis_functions())
            ]
        )
        local_to_global = mesh.connectivitymatrix[i]
        load_vector[np.ix_(local_to_global)] += local_load_vector
    load_vector = add_dirichlet_data(load_vector, dirichlet_data, mesh)
    load_vector = add_neumann_data(load_vector, neumann_data, mesh)
    return load_vector


@timing
def compute_loadvector_quadrature(
    rhs: Callable,
    basis_functions: BasisFunctions,
    dirichlet_data: Callable,
    neumann_data: Callable,
    mesh: Mesh,
    quadrature_rule: QuadratureRule,
) -> np.array:
    logger.info("Compute load vector")
    load_vector = np.zeros(shape=(mesh.number_of_nodes,))
    quadrature_matrix = np.array(
        [
            basis_functions.local_basis_functions(*quadrature_point).flatten()
            for quadrature_point in quadrature_rule.points
        ]
    ).T
    p1_vec = mesh.pointmatrix[:, 0, :]
    transformed_points_vec = np.transpose(
        p1_vec[..., None] + np.matmul(mesh.transformationmatrix, quadrature_rule.points.T), axes=(0, 2, 1)
    )
    rhs_quadrature_points_vec = rhs(transformed_points_vec[:, :, 0], transformed_points_vec[:, :, 1])
    basis_times_rhs = np.transpose(quadrature_matrix[..., None] * rhs_quadrature_points_vec.T, axes=(2, 0, 1))
    load_vector = mesh.determinant[..., None] * np.matmul(basis_times_rhs, quadrature_rule.weights)
    load_vector = coo_array(
        (
            load_vector.flatten(),
            (mesh.connectivitymatrix.ravel(), np.zeros_like(mesh.connectivitymatrix.ravel())),
        )
    )
    load_vector = np.squeeze(load_vector.toarray())
    load_vector = add_dirichlet_data(load_vector, dirichlet_data, mesh)
    load_vector = add_neumann_data(load_vector, neumann_data, mesh)
    return load_vector


def add_dirichlet_data(
    load_vector: np.array,
    dirichlet_data: Callable,
    mesh: Mesh,
) -> np.array:
    for index in mesh.boundary_indices:
        load_vector[index] = dirichlet_data(*mesh.nodes[index])
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
