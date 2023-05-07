def plot_solution(simulator):
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri

    # Create the matplotlib Triangulation object
    tri = simulator.mesh.connectivitymatrix[:, 0:3]  # or tess.simplices depending on scipy version
    triang = mtri.Triangulation(
        x=simulator.mesh.nodes[:, 0],
        y=simulator.mesh.nodes[:, 1],
        triangles=tri,
    )
    plt.triplot(
        simulator.mesh.nodes[:, 0],
        simulator.mesh.nodes[:, 1],
        simulator.mesh.connectivitymatrix[:, 0:3],
    )
    plt.plot(
        simulator.mesh.nodes[:, 0],
        simulator.mesh.nodes[:, 1],
        "o",
    )

    # Plotting
    fig = plt.figure()
    ax = fig.gca(projection="3d")
    my_cmap = plt.get_cmap("viridis")
    plot = ax.plot_trisurf(
        triang,
        simulator.solution.flatten(),
        cmap=my_cmap,
    )
    fig.colorbar(plot, ax=ax, shrink=0.5, aspect=5)
    plt.show()
