from ramanujantools.cmf.known_cmfs import pFq
import sympy as sp
# import pandas as pd
from trajectoryGenerator import get_trajectory_vector

x0, x1, y0 = sp.symbols('x0 x1 y0')
piCMF = pFq(2, 1, sp.Rational(1, 2))


def generate_limits_by_location(start, iterations=1000, d=10):
    angles = [i for i in range(0, 90 + 1, d)]
    limits = []
    directions = {'trajectory': [], 'theta': [], 'phi': []}

    for phi in angles:
        for theta in angles:
            directions['trajectory'].append(get_trajectory_vector(theta_deg=theta, phi_deg=phi))
            directions['theta'].append(theta)
            directions['phi'].append(phi)

    for i in range(len(directions['trajectory'])):
        dx, dy, dz = directions['trajectory'][i]
        direction = {x0: dx, x1: dy, y0: dz}
        try:
            lim = piCMF.limit(direction, iterations, start)
            deltas = piCMF.delta_sequence(direction, iterations, start)
            limits.append({
                'trajectory': direction,
                'limit': lim.as_float(),
                'theta': directions['theta'][i],
                'phi': directions['phi'][i],
                'delta': deltas
            })
        except:
            continue
    return limits


def generate_all_data(cube_factor=0.5, cube_size=2, iterations=1000):
    data = {
        'limit': [],
        'theta': [],
        'phi': [],
        'trajectory': [],
        'start': [],
        'delta': []
    }

    steps = [i * cube_factor for i in range(0, cube_size * int(1 / cube_factor) + 1)]
    print(steps)
    for a in steps:
        for b in steps:
            for c in steps:
                start = {x0: a if isinstance(a, int) else sp.nsimplify(a, rational=True),
                         x1: b if isinstance(b, int) else sp.nsimplify(b, rational=True),
                         y0: c if isinstance(c, int) else sp.nsimplify(c, rational=True)}
                limits = generate_limits_by_location(start, iterations)
                for limit in limits:
                    data['limit'].append(limit['limit'])
                    data['phi'].append(limit['phi'])
                    data['theta'].append(limit['theta'])
                    data['trajectory'].append(limit['trajectory'])
                    data['start'].append((a, b, c))
                    data['delta'].append(limit['delta'])
    return data


# def sort_by_trajectory(data):
#     by_trajectory = {trajectory: {'limit': [], 'phi': 0, 'theta': 0, 'start': []} for trajectory in set(data['trajectory'])}
#
#     for frame in data:
#         by_trajectory[frame['trajectory']]['limit'].append(frame['limit'])
#         by_trajectory[frame['trajectory']]['phi'] = frame['phi']
#         by_trajectory[frame['trajectory']]['theta'] = frame['theta']
#         by_trajectory[frame['trajectory']]['start'].append(frame['start'])
#     return by_trajectory


def sort_by_start(data):
    by_start = {start: {'limit': [], 'phi': [], 'theta': [], 'trajectory': [], 'delta': []} for start in set(data['start'])}
    
    for i in range(len(data['limit'])):
        frame = {
            'start': data['start'][i],
            'limit': data['limit'][i],
            'phi': data['phi'][i],
            'theta': data['theta'][i],
            'trajectory': data['trajectory'][i],
            'delta': data['delta'][i]
        }

        by_start[frame['start']]['limit'].append(frame['limit'])
        by_start[frame['start']]['phi'].append(frame['phi'])
        by_start[frame['start']]['theta'].append(frame['theta'])
        by_start[frame['start']]['trajectory'].append(frame['trajectory'])
        by_start[frame['start']]['delta'].append(frame['delta'])
    return by_start