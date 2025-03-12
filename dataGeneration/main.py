import dataGeneration.visual
from dataGeneration.visual import *
from dataGeneration.generator import *
from dataGeneration.format import *


def cubic_space(cube_factor=1, cube_size=1):
    steps = [i * cube_factor for i in range(0, cube_size * int(1 / cube_factor) + 1)]
    starting_points = [(x, y, z) for x in steps for y in steps for z in steps]
    return starting_points


def acquire_data(ts, te, ps, pe, step1, step2):
    data = from_json()
    if data:
        return data

    angle_range = generate_angle_range(ts, te, ps, pe, step1, step2)
    trajectories = generate_trajectories(angle_range)
    iterations = 5

    generation_data = [
        {
            'start': s,
            'trajectory': {
                'phi': phi,
                'theta': theta,
                'vector': vector
            },
            'limit': '',
            'depth': iterations
        }
        for phi, theta, vector in zip(trajectories['phi'], trajectories['theta'], trajectories['vector'])
        for s in cubic_space(cube_factor=1, cube_size=1)]
    return generation_data


def main():
    generation_data = acquire_data(0, 30, 0, 30, 10, 10)

    data = generate_delta_sequence(generation_data)

    filtered = extract_by_trajectory({'phi': 10, 'theta': 10}, data)
    filtered = extract_by_start((1, 0, 0), filtered)
    if filtered and filtered[0]['del_seq_unknown']:
        deltas = filtered[0]['del_seq_unknown']
        dataGeneration.visual.snip_delta_vs_depth_along_trajectory(deltas)
    to_json({'data': generation_data})


if __name__ == '__main__':
    main()
