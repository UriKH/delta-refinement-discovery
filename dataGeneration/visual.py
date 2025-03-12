import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  # Use interactive backend like TkAgg


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


def snip_delta_at_depth_vs_trajectory():
    pass


def snip_value_vs_trajectory():
    pass



"""
    Checks:
    * delta vs depth alongside a trajectory (limit known and unknown) (2D)
    * convergence vs angles (3D)
    * convergence of 1 trajectory vs start points (3D colored)
"""
