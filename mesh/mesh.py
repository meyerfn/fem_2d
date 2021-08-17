from scipy.spatial import Delaunay
import numpy as np


class Mesh:
    def __init__(self, nodes: np.array, neumann_edges: np.array) -> None:
        self.nodes = nodes
        self.number_of_nodes = len(nodes)
        triangulation = Delaunay(nodes, furthest_site=False)
        self.number_of_elements = len(triangulation.simplices)
        self.connectivitymatrix = triangulation.simplices
        self.pointmatrix = nodes[self.connectivitymatrix]
        self.__determine_boundary_indices_and_neumann_edges(triangulation, neumann_edges)
        self.__jacobian_and_determinant()

    def __jacobian_and_determinant(self) -> None:
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

    def __determine_boundary_indices_and_neumann_edges(
        self, triangulation: Delaunay, neumann_edges: np.array
    ) -> None:
        self.boundary_indices = set()
        self.neumann_edges = []
        for i, _ in enumerate(triangulation.neighbors):
            for k in range(3):
                if triangulation.neighbors[i][k] == -1:
                    index_one, index_two = (
                        (k + 1) % 3,
                        (k + 2) % 3,
                    )
                    vertex_one = triangulation.simplices[i][index_one]
                    vertex_two = triangulation.simplices[i][index_two]
                    self.boundary_indices.add(vertex_one)
                    self.boundary_indices.add(vertex_two)
                    if self.__is_edge_on_neumann_boundary(
                        self.nodes[vertex_one],
                        self.nodes[vertex_two],
                        neumann_edges,
                    ):
                        self.neumann_edges.append(
                            [vertex_one, vertex_two],
                        )
        self.neumann_edges = np.asarray(self.neumann_edges)
        all_nodes = set(range(self.number_of_nodes))
        self.free_indices = list(all_nodes.difference(self.boundary_indices))

    def __is_edge_on_neumann_boundary(
        self, vertex_one: np.array, vertex_two: np.array, neumann_edges: np.array
    ) -> bool:
        edge_on_neumann_edge = False
        for neumann_edge in neumann_edges:
            (
                vector_neumann_one_neumann_two,
                vector_vertex_one_neumann_one,
                vector_vertex_two_neumann_one,
            ) = self.__setup_vectors(neumann_edge, vertex_one, vertex_two)
            vertex_one_on_edge = np.cross(vector_neumann_one_neumann_two, vector_vertex_one_neumann_one) == 0
            vertex_two_on_edge = np.cross(vector_neumann_one_neumann_two, vector_vertex_two_neumann_one) == 0
            edge_on_neumann_edge = vertex_one_on_edge and vertex_two_on_edge
            if edge_on_neumann_edge:
                break
        return edge_on_neumann_edge

    def __setup_vectors(self, neumann_edge, vertex_one, vertex_two):
        neumann_vertex_one = np.array(neumann_edge[0, :])
        neumann_vertex_two = np.array(neumann_edge[1, :])
        vector_neumann_one_neumann_two = neumann_vertex_two - neumann_vertex_one
        vector_vertex_one_neumann_one = vertex_one - neumann_vertex_one
        vector_vertex_two_neumann_one = vertex_two - neumann_vertex_one
        return vector_neumann_one_neumann_two, vector_vertex_one_neumann_one, vector_vertex_two_neumann_one
