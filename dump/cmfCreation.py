from ramanujantools.cmf.known_cmfs import pFq
import sympy as sp
from trajectoryGenerator import get_trajectory_vector

x0, x1, y0 = sp.symbols('x0 x1 y0')
piCMF = pFq(2, 1, sp.Rational(1, 2))


def main():
    direction = {x0: 1, x1: 1, y0: 0}
    start = {x0: sp.Rational(1, 2), x1: sp.Rational(1, 2), y0: sp.Rational(3, 2)}

    angles = [i for i in range(0, 90, 10)]
    limits = []
    directions = set()
    for phi in angles:
        for theta in angles:
            directions.add(get_trajectory_vector(theta_deg=theta, phi_deg=phi))

    for dx, dy, dz in directions:
        direction = {x0: dx, x1: dy, y0: dz}
        # trajectory: Matrix = piCMF.trajectory_matrix(direction, start)
        # print(trajectory)
        # print(trajectory.as_pcf().pcf)
        try:
            lim = piCMF.limit(direction, 1000, start)
            print(f'({dx}, {dy}, {dz}): \t {lim.as_float()}')
        except:
            continue

    # trajectory: Matrix = piCMF.trajectory_matrix(direction, start)
    # print(trajectory)
    # print(trajectory.as_pcf().pcf)

    # lim = piCMF.limit(direction, 10000, start)
    # # print(pi().matrices)
    # # lim: list = pi().limit({x: 1, y: 1}, 10, {x: 0, y: 0})
    # print(lim.as_float())
    # print(lim)


if __name__ == '__main__':
    main()
