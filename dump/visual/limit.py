import numpy as np
import matplotlib.pyplot as plt


def show_limit_vs_angles():
    # Define the function f(theta, phi). You can change this function to whatever you need.
    def f(theta, phi):
        return np.sin(theta) * np.cos(phi)

    # Create a grid of theta and phi values.
    theta_vals = np.linspace(0, 2 * np.pi, 100)  # theta from 0 to 2*pi
    phi_vals = np.linspace(0, np.pi, 100)  # phi from 0 to pi
    theta, phi = np.meshgrid(theta_vals, phi_vals)

    # Evaluate the function on the grid.
    values = f(theta, phi)

    # Create the 3D plot.
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the surface.
    surf = ax.plot_surface(theta, phi, values, cmap='viridis', edgecolor='none')
    ax.set_xlabel('theta')
    ax.set_ylabel('phi')
    ax.set_zlabel('value')

    # Optionally, add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()

