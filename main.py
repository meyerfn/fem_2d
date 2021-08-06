from mesh.mesh import Mesh
from simulator.simulator import Simulator

import itertools
import numpy as np


def setUpNodes(number_of_nodes_1d: int) -> None:
    x = np.linspace(0, 1, number_of_nodes_1d, endpoint=True)
    nodes = []
    for r in itertools.chain(itertools.product(x, x)):
        nodes.append(r)
    return np.asarray(nodes)


# def rhs(node):
#     return -6.0


def rhs(node):
    return 1.0


# def dirichlet_data(node):
#     return 1 + node[0] ** 2 + 2.0 * node[1] ** 2


def dirichlet_data(node):
    return 0.0


if __name__ == "__main__":
    x = np.linspace(0, 1, 10, endpoint=True)
    nodes = []
    for r in itertools.chain(itertools.product(x, x)):
        nodes.append(r)
    nodes = np.asarray(nodes)
    mesh = Mesh(nodes)
    simulator = Simulator(mesh, dirichlet_data, rhs)
    simulator.simulate()

    import matplotlib.tri as mtri
    import matplotlib.pyplot as plt

    # Create the matplotlib Triangulation object
    x = mesh.pointmatrix[:, 0]
    y = mesh.pointmatrix[:, 1]
    tri = mesh.connectivitymatrix  # or tess.simplices depending on scipy version
    triang = mtri.Triangulation(
        x=nodes[:, 0],
        y=nodes[:, 1],
        triangles=tri,
    )

    # Plotting
    fig = plt.figure()
    ax = fig.gca(projection="3d")
    my_cmap = plt.get_cmap("viridis")
    plot = ax.plot_trisurf(
        triang,
        simulator.solution[:, 0],
        cmap=my_cmap,
    )
    fig.colorbar(plot, ax=ax, shrink=0.5, aspect=5)
    plt.show()
