from ramanujantools.cmf.known_cmfs import pFq
import sympy as sp
from trajectoryGenerator import get_trajectory_vector

import matplotlib.pyplot as plt
import numpy as np

# from mpl_toolkits.mplot3d import Axes3D


x0, x1, y0 = sp.symbols('x0 x1 y0')
piCMF = pFq(2, 1, sp.Rational(1, 2))


def main():
    # direction = {x0: 1, x1: 1, y0: 0}
    start = {x0: sp.Rational(1, 2), x1: sp.Rational(1, 2), y0: sp.Rational(3, 2)}
    angle_factor = 1
    angles = [i * angle_factor for i in range(0, 90 * int(1 / angle_factor), 1)]
    # limits = []
    directions = set()
    for phi in angles:
        for theta in angles:
            directions.add(get_trajectory_vector(theta_deg=theta, phi_deg=phi))

    print(directions)

    # for dx, dy, dz in directions:
    #     direction = {x0: dx, x1: dy, y0: dz}
    #     try:
    #         lim = piCMF.limit(direction, 10, start)
    #         print(f'({dx}, {dy}, {dz}): \t {lim.as_float()}')
    #     except Exception as e:
    #         # print(e)
    #         continue

    plot_vectors(convert_vectors_to_int(directions))


def normalize_vector(v):
    norm = np.linalg.norm(v)
    return v / norm if norm != 0 else v


def plot_vectors(vectors):
    def plot_vectors_helper(ax, vectors, show_vectors=True, show_tips=True, normalize=False, title=""):
        # Convert to numpy array for float handling
        vectors = np.array(vectors, dtype=float)

        if normalize:
            vectors = np.array([normalize_vector(v) for v in vectors])

        # Extract x, y, z coordinates
        x, y, z = vectors[:, 0], vectors[:, 1], vectors[:, 2]

        # Plot vector tips
        if show_tips:
            ax.scatter(x, y, z, color='red', label=f'Trajectory {"direction" if normalize else "step"}')

            # Plot vectors from the origin
            if show_vectors:
                for vx, vy, vz in vectors:
                    ax.quiver(0, 0, 0, vx, vy, vz, color='blue', arrow_length_ratio=0.1)

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

        plot_vectors_helper(axes[0, 0], vectors, show_vectors=False, show_tips=True, normalize=True,
                            title='Normalized - Tips')
        plot_vectors_helper(axes[0, 1], vectors, show_vectors=False, show_tips=True, normalize=False,
                            title='Original - Tips')
        plot_vectors_helper(axes[1, 0], vectors, show_vectors=True, show_tips=True, normalize=True,
                            title='Normalized - Vectors')
        plot_vectors_helper(axes[1, 1], vectors, show_vectors=True, show_tips=True, normalize=False,
                            title='Original - Vectors')

        plt.tight_layout()
        plt.show()

def convert_vectors_to_int(vectors):
    converted = [(int(x), int(y), int(z)) for x, y, z in vectors]
    return converted

if __name__ == '__main__':
    # Example usage
    main()
