def plot_solution(simulator):
    import matplotlib.tri as mtri
    import matplotlib.pyplot as plt

    # Create the matplotlib Triangulation object
    tri = simulator.mesh.connectivitymatrix  # or tess.simplices depending on scipy version
    triang = mtri.Triangulation(
        x=simulator.mesh.nodes[:, 0],
        y=simulator.mesh.nodes[:, 1],
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
