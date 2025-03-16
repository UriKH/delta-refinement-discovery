from cmfCreator import piCMF, x0, x1, y0
import sympy as sp
from typing import Dict
from format import *
from tqdm import tqdm


def generate_angle_range(t_start, t_end, p_start, p_end, dt=5, dp=5):
    directions = {'theta': [], 'phi': []}

    for phi in range(t_start, t_end + 1, dt):
        for theta in range(p_start, p_end + 1, dp):
            directions['theta'].append(theta)
            directions['phi'].append(phi)
    return directions


def cubic_space(
        start: float = 0,
        shift_x: float = 0,
        shift_y: float = 0,
        shift_z: float = 0,
        cube_factor: float = 1,
        cube_size: float = 1):
    steps = [start + i * cube_factor for i in range(int(cube_size / cube_factor) + 1)]
    starting_points = [(x+shift_x, y+shift_y, z+shift_z) for x in steps for y in steps for z in steps]
    return starting_points


def generate_trajectories_from_angles(angle_range):
    angle_range['vector'] = []

    for theta, phi in zip(angle_range['theta'], angle_range['phi']):
        vector = get_trajectory_vector(theta, phi)
        angle_range['vector'].append(vector)
    return angle_range


def get_rational_approx(expr, tolerance=1e-6, denom_limit=3):
    """
    Given a sympy expression 'expr', return the denominator of its rational
    approximation using nsimplify.
    """
    approx_expr = sp.nsimplify(expr.evalf(), rational=True, tolerance=tolerance).limit_denominator(denom_limit)
    return approx_expr


def get_trajectory_vector(theta_deg, phi_deg, tolerance=1e-6):
    """
    Given spherical angles theta (azimuthal) and phi (inclination) in degrees,
    returns the smallest integer vector (x, y, z) that points in the given direction.
    """
    theta = sp.rad(theta_deg)
    phi = sp.rad(phi_deg)

    # Direction vector components in spherical coordinates:
    # Here we use the convention: x = cos(theta)*sin(phi), y = sin(theta)*sin(phi), z = cos(phi)
    x_expr = sp.cos(theta) * sp.sin(phi)
    y_expr = sp.sin(theta) * sp.sin(phi)
    z_expr = sp.cos(phi)

    # Get the denominators of the rational approximations
    x = get_rational_approx(x_expr, tolerance)
    y = get_rational_approx(y_expr, tolerance)
    z = get_rational_approx(z_expr, tolerance)

    # Compute the least common multiple of the denominators
    lcm = sp.ilcm(x.as_numer_denom()[1], sp.ilcm(y.as_numer_denom()[1], z.as_numer_denom()[1]))

    # Return the integer vector
    return sp.simplify(lcm * x), sp.simplify(lcm * y), sp.simplify(lcm * z)


def create_start_vector(start: (float, float, float)):
    return {
        x0: start[0] if isinstance(start[0], int) else sp.nsimplify(start[0], rational=True),
        x1: start[1] if isinstance(start[1], int) else sp.nsimplify(start[1], rational=True),
        y0: start[2] if isinstance(start[2], int) else sp.nsimplify(start[2], rational=True)
    }


def generate_delta_sequence(generation_data, cmf_deltas=True, pcf_deltas=False):
    """
    :param generation_data: [{
            'start': <>,
            'trajectory': {
                'phi': x,
                'theta': x,
                'vector': <>
            },
            'limit': None,
            'depth': n}
        ... ]
    """
    for i, gen_data in tqdm(enumerate(generation_data), desc="calculating delta sequences", total=len(generation_data)):
        start = create_start_vector(gen_data['start'])
        trajectory_vec = gen_data['trajectory']

        deltas_known_cmf_limit = []
        deltas_unknown_cmf_limit = []
        deltas_known_pcf_limit = []
        deltas_unknown_pcf_limit = []

        try:
            if cmf_deltas:
                if gen_data['cmf_known_limit']:
                    deltas_known_cmf_limit = piCMF.delta_sequence(
                        trajectory={x0: trajectory_vec[0], x1: trajectory_vec[1], y0: trajectory_vec[2]},
                        depth=gen_data['depth'],
                        start=start,
                        limit=gen_data['cmf_known_limit']
                    )

                deltas_unknown_cmf_limit = piCMF.delta_sequence(
                    trajectory={x0: trajectory_vec[0], x1: trajectory_vec[1], y0: trajectory_vec[2]},
                    depth=gen_data['depth'],
                    start=start
                )

            if pcf_deltas:
                trajectory_mat_pcf = piCMF.trajectory_matrix().as_pcf().pcf
                if gen_data['pcf_known_limit']:
                    deltas_known_pcf_limit = trajectory_mat_pcf.delta_sequence(gen_data['depth'], gen_data['pcf_known_limit'])
                deltas_unknown_pcf_limit = trajectory_mat_pcf.delta_sequence(gen_data['depth'])
        except:
            pass
        finally:
            generation_data[i]['deltas_known_cmf_limit'] = [float(delta) for delta in deltas_known_cmf_limit]
            generation_data[i]['deltas_unknown_cmf_limit'] = [float(delta) for delta in deltas_unknown_cmf_limit]
            generation_data[i]['deltas_known_pcf_limit'] = deltas_known_pcf_limit
            generation_data[i]['deltas_unknown_pcf_limit'] = deltas_unknown_pcf_limit
    return generation_data


def generate_convergence(generation_data, cmf_conv=True, pcf_conv=False):
    for i, gen_data in tqdm(enumerate(generation_data), desc="calculating limits", total=len(generation_data)):
        start = create_start_vector(gen_data['start'])
        trajectory_vec = gen_data['trajectory']

        cmf_limit = None
        pcf_limit = None

        try:
            if cmf_conv:
                cmf_limit = piCMF.limit(
                    trajectory={x0: trajectory_vec[0], x1: trajectory_vec[1], y0: trajectory_vec[2]},
                    iterations=gen_data['depth'],
                    start=start
                ).as_float()
            if pcf_conv:
                pcf_limit = piCMF.trajectory_matrix().as_pcf().pcf.limit(gen_data['depth']).as_float()
        except:
            pass
        finally:
            generation_data[i]['cmf_limit'] = cmf_limit
            generation_data[i]['pcf_limit'] = pcf_limit
    return generation_data


def extract_by_trajectory(t, data, by_vector_x=False, by_vector_y=False, by_vector_z=False):
    filtered = []
    for frame in data:
        if by_vector_x and frame['trajectory'][0] != t[0]:
            continue
        if by_vector_y and frame['trajectory'][1] != t[1]:
            continue
        if by_vector_z and frame['trajectory'][2] != t[2]:
            continue
        filtered.append(frame)
    return filtered


def extract_by_start(s, data, by_x=True, by_y=True, by_z=True):
    filtered = []
    for frame in data:
        if by_x and frame['start'][0] != s[0]:
            continue
        if by_y and frame['start'][1] != s[1]:
            continue
        if by_z and frame['start'][2] != s[2]:
            continue
        filtered.append(frame)
    return filtered


def acquire_data(start_space,
                 generate_from_angles=True,
                 iterations=5,
                 use_json=False,
                 from_cubic_space: Dict[str, int | float] = None,
                 angles: Dict[str, int | float] = None):
    if use_json:
        data = from_json()
        if data:
            return data

    if generate_from_angles:
        angle_range = generate_angle_range(**angles)
        trajectories = generate_trajectories_from_angles(angle_range)
    else:
        coord_space = cubic_space(**from_cubic_space)
        trajectories = {'vector': coord_space}

    generation_data = [
        {
            'start': s,
            'trajectory': vector,
            'pcf_known_limit': '',
            'cmf_known_limit': '',
            'depth': iterations
        }
        for vector in trajectories['vector']
        for s in start_space]
    return generation_data
