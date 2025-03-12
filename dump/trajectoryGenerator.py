import sympy as sp


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