"""
functions module.
"""
import numpy as np
from sympy import lambdify, abc, latex
from sympy.parsing.sympy_parser import parse_expr


def rect(x: np.ndarray) -> np.ndarray:
    """
    Rectangle function.
    """
    try:
        return np.array(
            [
                1.0 if (x_i < 0.5 and x_i > -0.5) else 0.
                for x_i in x
            ]
            )
    except:
        return 1.0 if (x < 0.5 and x > -0.5) else 0.


def delta(x: np.ndarray) -> np.ndarray:
    """
    Discrete approximation of the dirac delta function.
    """
    try:
        dx = (x[-1] - x[0])/len(x)
        return np.array([1e10 if (xi < (0. + dx/2) and xi > (0. - dx/2))
                         else 0. for xi in x])
    except:
        return 1e10 if (x < 0.01 and x > -0.01) \
               else 0.


# Dictionary of modules and user defined functions.
# Used for lambdify from sympy to parse input.
module_list = ["numpy", {"rect":rect}]


def convert_to_function(string: str, scale_by_k=False):
    """Using the sympy module, parse string input
    into a mathematical expression.
    Returns the original string, the latexified string,
    the mathematical expression in terms of sympy symbols,
    and a lambdified function
    """
    string = string.replace("^", "**")
    symbolic_function = parse_expr(string)
    if scale_by_k:
        latexstring = latex(symbolic_function*abc.k)
    else:
        latexstring = latex(symbolic_function)
    lambda_function = lambdify(abc.x, symbolic_function,
                               modules=module_list)
    string = string.replace('*', '')
    latexstring = "$" + latexstring + "$"
    return string, latexstring, \
           symbolic_function, lambda_function

