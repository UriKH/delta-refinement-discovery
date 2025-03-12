from cmfCreator import piCMF, x0, x1, y0
import sympy as sp


def generate_angle_range(start1, end1, start2, end2, delta1=5, delta2=5):
    directions = {'theta': [], 'phi': []}

    for phi in range(start1, end1 + 1, delta1):
        for theta in range(start2, end2 + 1, delta2):
            directions['theta'].append(theta)
            directions['phi'].append(phi)
    return directions


def generate_trajectories(angle_range):
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

def generate_delta_sequence(generation_data):
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
    for i, gen_data in enumerate(generation_data):
        start = create_start_vector(gen_data['start'])
        trajectory_vec = gen_data['trajectory']['vector']

        deltas_known_limit = []
        deltas_unknown_limit = []

        try:
            if gen_data['limit']:
                deltas_known_limit = piCMF.delta_sequence(
                    trajectory={x0: trajectory_vec[0], x1: trajectory_vec[1], y0: trajectory_vec[2]},
                    depth=gen_data['depth'],
                    start=start,
                    limit=gen_data['limit']
                )

            deltas_unknown_limit = piCMF.delta_sequence(
                trajectory={x0: trajectory_vec[0], x1: trajectory_vec[1], y0: trajectory_vec[2]},
                depth=gen_data['depth'],
                start=start
            )

        except:
            continue
        finally:
            generation_data[i]['del_seq_known'] = [float(delta) for delta in deltas_known_limit]
            generation_data[i]['del_seq_unknown'] = [float(delta) for delta in deltas_unknown_limit]
    return generation_data


def generate_convergence(generation_data):
    for i, gen_data in enumerate(generation_data):
        start = create_start_vector(gen_data['start'])
        trajectory_vec = gen_data['trajectory']['vector']

        try:
            limit = piCMF.limit(
                trajectory={x0: trajectory_vec[0], x1: trajectory_vec[1], y0: trajectory_vec[2]},
                iterations=gen_data['depth'],
                start=start
            )
            generation_data[i]['limit'] = limit.as_float()
        except:
            continue
    return generation_data


def extract_by_trajectory(t, data, by_phi=True, by_theta=True, by_vector_x=False, by_vector_y=False, by_vector_z=False):
    filtered = []
    for frame in data:
        if by_phi and frame['trajectory']['phi'] != t['phi']:
            continue
        if by_theta and frame['trajectory']['theta'] != t['theta']:
            continue
        if by_vector_x and frame['trajectory']['vector'][0] != t['vector'][0]:
            continue
        if by_vector_y and frame['trajectory']['vector'][1] != t['vector'][1]:
            continue
        if by_vector_z and frame['trajectory']['vector'][2] != t['vector'][2]:
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
