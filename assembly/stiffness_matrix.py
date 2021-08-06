from mesh.mesh import Mesh
import numpy as np

local_linear_dx_dx = np.array(
    [
        [0.5, -0.5, 0.0],
        [-0.5, 0.5, 0.0],
        [0.0, 0.0, 0.0],
    ]
)
local_linear_dx_dy = np.array(
    [
        [0.5, 0.0, -0.5],
        [-0.5, 0.0, 0.5],
        [0.0, 0.0, 0.0],
    ]
)
local_linear_dy_dy = np.array(
    [
        [0.5, 0.0, -0.5],
        [0.0, 0.0, 0.0],
        [-0.5, 0.0, 0.5],
    ]
)


def compute_stiffnessmatrix(
    mesh: Mesh,
) -> np.array:
    stiffness_matrix = np.zeros(
        shape=(
            mesh.number_of_nodes,
            mesh.number_of_nodes,
        )
    )
    for i in range(mesh.number_of_elements):
        local_stiffnessmatrix = compute_local_stiffnessmatrix(mesh, i)
        local_to_global = mesh.connectivitymatrix[i]
        stiffness_matrix[np.ix_(local_to_global, local_to_global)] += local_stiffnessmatrix
    stiffness_matrix = remove_dirichlet_nodes(stiffness_matrix, mesh.boundary_indices)
    return stiffness_matrix


def remove_dirichlet_nodes(
    stiffness_matrix: np.array,
    boundary_indices: list,
) -> np.array:
    for row in boundary_indices:
        stiffness_matrix[row, :] = np.eye(1, len(stiffness_matrix), row)
    return stiffness_matrix


def compute_local_coordinate_transformation(
    jacobian_matrix: np.array,
) -> np.array:
    return np.linalg.inv(jacobian_matrix) * np.linalg.inv(np.transpose(jacobian_matrix))


def compute_local_stiffnessmatrix(mesh: Mesh, index: int) -> np.array:
    b, c = local_shape_coefficients(
        mesh.pointmatrix[index],
        mesh.determinant[index],
    )
    local_stiffness_matrix = (np.outer(b, b) + np.outer(c, c)) * 0.5 * mesh.determinant[index]
    return local_stiffness_matrix


def local_shape_coefficients(nodes: np.array, determinant: float):
    x = nodes[:, 0]
    y = nodes[:, 1]
    area = 0.5 * determinant
    b = (
        np.array(
            [
                [y[1] - y[2]],
                [y[2] - y[0]],
                [y[0] - y[1]],
            ]
        )
        / 2.0
        / area
    )
    c = (
        np.array(
            [
                [x[2] - x[1]],
                [x[0] - x[2]],
                [x[1] - x[0]],
            ]
        )
        / 2.0
        / area
    )
    return b, c


# def compute_local_stiffnessmatrix(mesh: Mesh, index: int) -> np.array:
#     jacobian = mesh.jacobian[index]
#     local_coordinate_transformation = compute_local_coordinate_transformation(jacobian)

#     local_stiffnessmatrix = np.linalg.det(jacobian) * (
#         local_coordinate_transformation[0, 0] * local_linear_dx_dx
#         + local_coordinate_transformation[1, 1] * local_linear_dy_dy
#         + local_coordinate_transformation[0, 1] * (local_linear_dx_dy + np.transpose(local_linear_dx_dy))
#     )
#     return local_stiffnessmatrix
