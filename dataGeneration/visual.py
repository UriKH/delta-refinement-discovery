import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  # Use interactive backend like TkAgg


def normalize_vector(v):
    norm = np.linalg.norm(v)
    return v / norm if norm != 0 else v


def plot_vectors(vectors, start_points=None):
    def plot_vectors_helper(ax, vectors, start_points=None, show_vectors=True, show_tips=True, normalize=False, title=""):
        # Convert to numpy array for float handling
        vectors = np.array(vectors, dtype=float)
        if not start_points:
            start_points = np.zeros(vectors.shape, dtype=float)
        start_points = np.array(start_points, dtype=float)

        if normalize:
            vectors = np.array([normalize_vector(v) for v in vectors])
            start_points = np.array([normalize_vector(v) for v in start_points])

        # Extract x, y, z coordinates
        x, y, z = vectors[:, 0], vectors[:, 1], vectors[:, 2]

        # Plot vector tips
        if show_tips:
            ax.scatter(x, y, z, color='red', label=f'Trajectory {"direction" if normalize else "step"}')

            # Plot vectors from the origin
            if show_vectors:
                for (sx, sy, sz), (vx, vy, vz) in zip(start_points, vectors):
                    ax.quiver(sx, sy, sz, vx, vy, vz, color='blue', arrow_length_ratio=0.1)

            # Labels and title
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title(title)

            # Set equal axis limits
            max_range = np.max(np.abs(vectors)) * 1.2  # Add some margin
            ax.set_xlim([-max_range, max_range])
            ax.set_ylim([-max_range, max_range])
            ax.set_zlim([-max_range, max_range])

    _, axes = plt.subplots(2, 2, subplot_kw={'projection': '3d'}, figsize=(10, 10))

    plot_vectors_helper(axes[0, 0], vectors, start_points, show_vectors=False, show_tips=True, normalize=True,
                        title='Normalized - Tips')
    plot_vectors_helper(axes[0, 1], vectors, start_points, show_vectors=False, show_tips=True, normalize=False,
                        title='Original - Tips')
    plot_vectors_helper(axes[1, 0], vectors, start_points, show_vectors=True, show_tips=True, normalize=True,
                        title='Normalized - Vectors')
    plot_vectors_helper(axes[1, 1], vectors, start_points, show_vectors=True, show_tips=True, normalize=False,
                        title='Original - Vectors')

    plt.tight_layout()
    plt.show()

def convert_vectors_to_int(vectors):
    converted = [(int(x), int(y), int(z)) for x, y, z in vectors]
    return converted


def snip_delta_vs_depth_along_trajectory(deltas, step_size=1, known_limit=None, with_known_limit=False):
    # Sample data (x, y points)
    x = np.array(range(1, step_size * len(deltas) + 1))
    scaled = []
    for delta in deltas:
        scaled += [delta for _ in range(step_size)]
    y = np.array(scaled)

    # Create the plot
    plt.figure(figsize=(8, 6))

    # Scatter plot for points
    plt.scatter(x, y, color='red', label='Points')

    # Line plot to connect the dots
    plt.plot(x, y, color='blue', linestyle='-', linewidth=1)

    # Labels and title
    plt.xlabel('iterations')
    plt.ylabel('delta')
    plt.title('delta vs iterations')
    plt.legend()

    # Show the plot
    plt.show()


def snip_convergence_vs_angles(thetas, phis, limits):
    theta_vals = np.array(thetas)  # theta from 0 to 2*pi
    phi_vals = np.array(phis)  # phi from 0 to pi
    values = np.array(limits)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot instead of surface plot
    sc = ax.scatter(theta_vals, phi_vals, values, c=values, cmap='viridis')

    ax.set_xlabel('Theta')
    ax.set_ylabel('Phi')
    ax.set_zlabel('Value')

    fig.colorbar(sc, shrink=0.5, aspect=5)
    plt.show()


def snip_value_vs_trajectory(project=False):
    # Example data (x, y, z coordinates and intensity values)
    x = np.random.rand(100)
    y = np.random.rand(100)
    z = np.random.rand(100)
    intensity = np.random.rand(100)  # Intensity values at each coordinate

    # Create a 3D scatter plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot with color based on intensity
    scatter = ax.scatter(x, y, z, c=intensity, cmap='viridis', s=50)

    # Add color bar to show intensity scale
    fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=10)

    # Grid customization (dotted lines)
    ax.grid(True, linestyle=':', color='gray', alpha=0.7)

    # Projections of points onto the xy, yz, and xz planes with uniform color (e.g., gray)
    projection_color = 'gray'  # Uniform color for projections

    if project:
        # Projection onto the XY plane (z=0)
        ax.scatter(x, y, np.zeros_like(z), c=projection_color, s=30, marker='o', alpha=0.5)

        # Projection onto the YZ plane (x=0)
        ax.scatter(np.zeros_like(x), y, z, c=projection_color, s=30, marker='o', alpha=0.5)

        # Projection onto the XZ plane (y=0)
        ax.scatter(x, np.zeros_like(y), z, c=projection_color, s=30, marker='o', alpha=0.5)

    # Labels and title
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.set_title('3D Scatter Plot with Dotted Grid and Uniform Colored Projections')

    # Show plot
    plt.show()

# snip_value_vs_trajectory()


"""
    Checks:
    * delta vs depth alongside a trajectory (limit known and unknown) (2D)
    * convergence vs angles (3D)
    * convergence of 1 trajectory vs start points (3D colored)
"""
