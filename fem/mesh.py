import numpy as np
from scipy.spatial import Delaunay
from fem.utils import timing


class Mesh:
    def __init__(self, nodes: np.array, neumann_edges: np.array) -> None:
        self.nodes = nodes
        triangulation = Delaunay(nodes, furthest_site=False)
        self.connectivitymatrix = triangulation.simplices
        self.pointmatrix = nodes[self.connectivitymatrix]
        self.neighbors = triangulation.neighbors
        self._jacobian_and_determinant()
        self._transformationmatrix()

    @property
    def number_of_nodes(self):
        return len(self.nodes)

    @property
    def number_of_elements(self):
        return len(self.connectivitymatrix)

    def _jacobian_and_determinant(self) -> None:
        self.jacobian = []
        self.determinant = []
        for elem in self.pointmatrix:
            local_jacobian = np.array(
                [
                    [
                        elem[1, 0] - elem[0, 0],
                        elem[2, 0] - elem[0, 0],
                    ],
                    [
                        elem[1, 1] - elem[0, 1],
                        elem[2, 1] - elem[0, 1],
                    ],
                ]
            )
            self.jacobian.append(local_jacobian)
            local_determinant = np.linalg.det(local_jacobian)
            self.determinant.append(local_determinant)
        self.determinant = np.array(self.determinant)

    def _transformationmatrix(self):
        p1 = self.pointmatrix[:, 0, :]
        p2 = self.pointmatrix[:, 1, :]
        p3 = self.pointmatrix[:, 2, :]
        self.transformationmatrix = np.transpose(np.array([p2 - p1, p3 - p1]), axes=(1, 2, 0))


class LinearMesh(Mesh):
    def __init__(self, nodes: np.array, neumann_edges: np.array) -> None:
        super().__init__(nodes, neumann_edges)
        self.boundary_indices, self.neumann_edges = determine_boundary_indices_and_neumann_edges(
            self.nodes, self.neighbors, self.connectivitymatrix, neumann_edges
        )


class QuadraticMesh(Mesh):
    def __init__(self, nodes: np.array, neumann_edges: np.array) -> None:
        super().__init__(nodes, neumann_edges)
        self._update_mesh_with_edge_points()
        self.boundary_indices, self.neumann_edges = determine_boundary_indices_and_neumann_edges(
            self.nodes, self.neighbors, self.connectivitymatrix, neumann_edges, add_midpoint=True
        )

    @timing
    def _update_mesh_with_edge_points(self) -> None:
        updated_point_matrix = np.zeros(
            shape=(self.pointmatrix.shape[0], self.pointmatrix.shape[1] + 3, self.pointmatrix.shape[2])
        )
        updated_connectivity_matrix = np.zeros(
            shape=(self.connectivitymatrix.shape[0], self.connectivitymatrix.shape[1] + 3), dtype=np.int32
        )
        for i, (elem, connectivity) in enumerate(zip(self.pointmatrix, self.connectivitymatrix)):
            midpoint_one = (elem[0] + elem[1]) / 2.0
            midpoint_two = (elem[0] + elem[2]) / 2.0
            midpoint_three = (elem[1] + elem[2]) / 2.0
            elem = np.vstack((elem, midpoint_one, midpoint_two, midpoint_three))
            for point in [midpoint_one, midpoint_two, midpoint_three]:
                if not (point == self.nodes).all(1).any():
                    self.nodes = np.vstack((self.nodes, point))
                connectivity = np.concatenate((connectivity, np.where((point == self.nodes).all(1))[-1]))
            updated_connectivity_matrix[i] = connectivity
            updated_point_matrix[i] = elem
        self.pointmatrix = updated_point_matrix
        self.connectivitymatrix = updated_connectivity_matrix


def determine_boundary_indices_and_neumann_edges(
    nodes: np.array,
    neighbors: np.array,
    connectivitymatrix: np.array,
    neumann_edges: np.array,
    add_midpoint=False,
) -> None:
    boundary_indices = set()
    neumann_edges = []
    for i, _ in enumerate(neighbors):
        for k in range(3):
            if neighbors[i][k] == -1:
                index_one, index_two = (
                    (k + 1) % 3,
                    (k + 2) % 3,
                )
                vertex_one = connectivitymatrix[i][index_one]
                vertex_two = connectivitymatrix[i][index_two]
                boundary_indices.add(vertex_one)
                boundary_indices.add(vertex_two)
                if add_midpoint:
                    index_midpoint = index_one + index_two + 2
                    vertex_midpoint = connectivitymatrix[i][index_midpoint]
                    boundary_indices.add(vertex_midpoint)

                if is_edge_on_neumann_boundary(
                    nodes[vertex_one],
                    nodes[vertex_two],
                    neumann_edges,
                ):
                    if add_midpoint:
                        neumann_edges.append(
                            [vertex_one, vertex_two, vertex_midpoint],
                        )
                    else:
                        neumann_edges.append(
                            [vertex_one, vertex_two],
                        )

    neumann_edges = np.asarray(neumann_edges)
    return boundary_indices, neumann_edges


def is_edge_on_neumann_boundary(vertex_one: np.array, vertex_two: np.array, neumann_edges: np.array) -> bool:
    edge_on_neumann_edge = False
    for neumann_edge in neumann_edges:
        (
            vector_neumann_one_neumann_two,
            vector_vertex_one_neumann_one,
            vector_vertex_two_neumann_one,
        ) = _setup_vectors(neumann_edge, vertex_one, vertex_two)
        vertex_one_on_edge = np.cross(vector_neumann_one_neumann_two, vector_vertex_one_neumann_one) == 0
        vertex_two_on_edge = np.cross(vector_neumann_one_neumann_two, vector_vertex_two_neumann_one) == 0
        edge_on_neumann_edge = vertex_one_on_edge and vertex_two_on_edge
        if edge_on_neumann_edge:
            break
    return edge_on_neumann_edge


def _setup_vectors(neumann_edge, vertex_one, vertex_two):
    neumann_vertex_one = np.array(neumann_edge[0, :])
    neumann_vertex_two = np.array(neumann_edge[1, :])
    vector_neumann_one_neumann_two = neumann_vertex_two - neumann_vertex_one
    vector_vertex_one_neumann_one = vertex_one - neumann_vertex_one
    vector_vertex_two_neumann_one = vertex_two - neumann_vertex_one
    return vector_neumann_one_neumann_two, vector_vertex_one_neumann_one, vector_vertex_two_neumann_one
