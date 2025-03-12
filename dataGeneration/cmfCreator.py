from ramanujantools.cmf.known_cmfs import pFq
import sympy as sp

x0, x1, y0 = sp.symbols('x0 x1 y0')
piCMF = pFq(2, 1, sp.Rational(1, 2))
