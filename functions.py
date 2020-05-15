"""
functions module.
"""
import numpy as np
from sympy import lambdify, abc, latex, diff, integrate
from sympy.parsing.sympy_parser import parse_expr
from sympy.core import basic
from typing import Dict, List, Union


class VariableNotFoundError(Exception):
    """Variable not found error.
    """
    def __str__(self) -> None:
        """Print this exception.
        """
        return "Variable not found"


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


def noise(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    This is the noise function.
    """
    if isinstance(x, np.ndarray):
        return np.array([2.0*np.random.rand() - 1.0 for _ in range(len(x))])
    else:
        return 2.0*np.random.rand() - 1.0


def multiplies_var(main_var: basic.Basic, arb_var: basic.Basic,
                   expr: basic.Basic) -> bool:
    """
    This function takes in the following parameters:
    main_var [sympy.core.basic.Basic]: the main variable
    arb_var [sympy.core.basic.Basic]: an arbitrary variable
    expr [sympy.core.basic.Basic]: an algebraic expression
    Check to see if an arbitrary variable multiplies
    a sub expression that contains the main variable.
    If it does, return True else False.
    """
    arg_list = []
    for arg1 in expr.args:
        if arg1.has(main_var):
            arg_list.append(arg1)
            for arg2 in expr.args:
                if ((arg2 is arb_var or (arg2.is_Pow and arg2.has(arb_var)))
                   and expr.has(arg1*arg2)):
                    return True
    return any([multiplies_var(main_var, arb_var, arg)
                for arg in arg_list if
                (arg is not main_var)])


class FunctionRtoR:
    """
    A callable function class that maps a single variable,
    as well as any number of parameters, to another variable.

    Attributes:
    latex_repr [str]: The function as a LaTeX string.
    symbols [sympy.Symbol]: All variables used in this function.
    parameters [sympy.Symbol]: All variables used in this function,
                               except for the main variable.
    """

    module_list = ["numpy", {"rect": rect, "noise": noise}]
    # Private Attributes:
    # _symbolic_func [sympy.basic.Basic]: symbol function
    # _lambda_func [sympy.Function]: lamba function

    def __init__(self, function_name: str,
                 param: Union[basic.Basic, str]) -> None:
        """
        The initializer. The parameter must be a
        string representation of a function, and it needs to
        be a function of x.
        """
        # Dictionary of modules and user defined functions.
        # Used for lambdify from sympy to parse input.
        if isinstance(param, str):
            param = parse_expr(param)
        self._symbolic_func = parse_expr(function_name)
        symbol_set = self._symbolic_func.free_symbols
        if abc.k in symbol_set:
            k_param = parse_expr("k_param")
            self._symbolic_func = self._symbolic_func.subs(abc.k, k_param)
            symbol_set = self._symbolic_func.free_symbols
        symbol_list = list(symbol_set)
        if param not in symbol_list:
            raise VariableNotFoundError
        self.latex_repr = latex(self._symbolic_func)
        symbol_list.remove(param)
        self.parameters = symbol_list
        var_list = [param]
        var_list.extend(symbol_list)
        self.symbols = var_list
        self._lambda_func = lambdify(
            self.symbols, self._symbolic_func, modules=self.module_list)

    def __call__(self, x: Union[np.array, float],
                 *args: float) -> np.array:
        """
        Call this class as if it were a function.
        """
        if args == ():
            kwargs = self.get_default_values()
            args = (kwargs[s] for s in kwargs)
        return self._lambda_func(x, *args)

    def __str__(self) -> str:
        """
        string representation of the function.
        """
        return str(self._symbolic_func)

    def multiply_latex_string(self, var: str) -> str:
        var = parse_expr(var)
        expr = var*self._symbolic_func
        return latex(expr)

    def _reset_samesymbols(self) -> None:
        """
        Set to a new function, assuming the same variables.
        """
        self.latex_repr = latex(self._symbolic_func)
        self._lambda_func = lambdify(
            self.symbols, self._symbolic_func)

    def get_default_values(self) -> Dict[basic.Basic, float]:
        """
        Get a dict of the suggested default values for each parameter
        used in this function.
        """
        return {s:
                float(multiplies_var(self.symbols[0], s, self._symbolic_func))
                for s in self.parameters}

    def get_enumerated_default_values(self) -> dict:
        """
        Get an enumerated dict of the suggested default values for each parameter
        used in this function.
        """
        return {i: [s, 
                float(multiplies_var(
                      self.symbols[0], s, self._symbolic_func))]
                for i, s in enumerate(self.parameters)}

    def get_tupled_default_values(self) -> tuple:
        """
        Get the suggested default values as a tuple.
        """
        enum_defaults = self.get_enumerated_default_values()
        return tuple([enum_defaults[i][1] for 
                      i in range(len(self.parameters))])

    @staticmethod
    def add_function(function_name, new_function) -> None:
        """
        """
        FunctionRtoR.module_list[1][function_name] = new_function

