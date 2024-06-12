import numpy as np
import sympy as sp
from sympy import sqrt, sin, cos, tan
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from typing import Any, Tuple
# !pip install ipympl
# !pip install mpl_interactions
# from google.colab import output
# output.enable_custom_widget_manager()
# from mpl_interactions import ioff, panhandler, zoom_factory
from IPython.display import display, Math
from ctypes import pointer
from math import isinf, isnan, sqrt
from typing import List
from sympy.simplify import nsimplify
from sympy import symbols, sympify, diff, factorial, limit, expand, factor, simplify
import warnings
## Extra packages
from sympy import sin, cos, tan, exp, ln, sqrt, pi, log, atan
from sympy import oo, zoo
from sympy.abc import x, n
import os

steps = []

# %matplotlib widget

def find_indet(function, sym, range_graph: Tuple[int, int]):
    """
    Find points where the function is indeterminate (division by zero) and compute the corresponding limits.

    Parameters:
    - function: The sympy function to analyze.
    - sym: The sym with respect to which the limit is computed.
    - range_graph: Tuple representing the range of x-values to analyze.

    Returns:
    - div_by_zero: List of x-values where the function is indeterminate.
    - lims: Dictionary containing limits computed at the indeterminate points.
    """
    div_by_zero = []
    lims = {}

    # Check if the function is a rational function
    if function.is_rational_function():
        denominator = function.as_numer_denom()[1]

        # Iterate over the range of x-values to find where the denominator becomes zero
        for x_val in range(range_graph[0], range_graph[1], 1):
            temp = denominator.subs(sym, x_val)
            if temp == 0:
                div_by_zero.append(x_val)

        # Compute the limits at the points of indeterminacy
        for i in range(0, len(div_by_zero)):
            lims[sp.limit(function, sym, div_by_zero[i])] = div_by_zero[i]
    else:
        div_by_zero = None

    return div_by_zero, lims

def dec(ax, p, l):
    """
    Decorate the plot with arrows and annotations at a specified point.

    Parameters:
    - ax: Axes object representing the plot.
    - p: x-coordinate of the point.
    - l: Limit value corresponding to the point.
    """
    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    ax.plot((1), (0), ls="", marker=">", ms=10, color="k",
            transform=ax.get_yaxis_transform(), clip_on=False)
    ax.plot((0), (1), ls="", marker="^", ms=10, color="k",
            transform=ax.get_xaxis_transform(), clip_on=False)

    plt.annotate(
        f'({p},{l})', xy=(p, float(l)), xytext=(p + 10, float(l) + 20),
        textcoords='offset points', arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.5'),
        path_effects=[pe.withStroke(linewidth=4, foreground="white")], zorder=20
        )

    plt.legend()
    plt.grid(True)

def isinf(l):
    """
    Check if a limit is infinity.

    Parameters:
    - l: The limit value to check.

    Returns:
    - True if the limit is infinity, else False.
    """
    return l in [sp.oo, -sp.oo, sp.zoo]

def plot_limit(
        function: Any,
        sym: sp.Symbol,
        index,
        point: int=None,
        range_x: Tuple[float | int, float | int] = None,
        range_y: Tuple[float | int, float | int] = None,
        range_graph: Tuple[int, int] = (-10,10),
        q: int = 500,
        annotate: bool = True
) -> int:
    """
    Plot the function and mark a specified point with its limit.

    Parameters:
    - function: The function to plot.
    - sym: The symbol with respect to which the limit is computed.
    - point: The x-value of the point to mark.
    - range_x: Tuple representing the x-axis limits of the plot.
    - range_y: Tuple representing the y-axis limits of the plot.
    - range_graph: Tuple representing the range of x-values to plot.
    - q: Number of points to generate in the plot.
    - annotate: Boolean indicating whether to annotate the plot with arrows and text.

    Returns:
    - 1 indicating the success of the plotting operation.
    """

    fig, ax = plt.subplots()

    # Find points of indeterminacy and compute the corresponding limits
    indet, lims = find_indet(function, sym, range_graph)

    # If a specific point is not provided, choose the first point of indeterminacy
    if point is None:
        if lims:
            point = list(lims.values())[0]
            limit = list(lims.keys())[0]
        # If no indeterminacy has been found, then just initialize
        else:
            point, limit = 0, 0
    # Use the provided point to calculate its limit
    else:
        limit = sp.limit(function, sym, point)

    # Draw the graph limits around the points if not provided
    if range_x is None:
        range_x = (-10, 10) if isinf(point) else (float(point) - 10, float(point) + 10)
    if range_y is None:
        range_y = (-10, 10) if isinf(point) else (float(point) - 10, float(point) + 10)

    # Generate x-values for plotting
    x_vals = np.linspace(*range_graph, int(q))
    x_vals = x_vals[~np.isin(x_vals, indet)] if indet is not None else x_vals
    y_vals = sp.lambdify(sym, function)(x_vals)

    # If the point is infinity, plot a horizontal line at the point of interest.
    if isinf(point):
        ax.axhline(
            limit,
            color='white',
            zorder=4
        )
        ax.axhline(
            limit,
            color='red',
            linestyle='--',
            zorder=10
        )

    # If the limit is infinity, plot a vertical line at the point of interest.
    # For the first axvline function, we plot a blank line at the point so we
    # can better visualize the actual infinity line.
    if indet:
        for indet_point in indet:
            ax.axvline(
                indet_point,
                color='white',
                zorder=4
            )
            ax.axvline(
                indet_point,
                color='red',
                linestyle='--',
                zorder=10
            )

    if not isinf(limit):
    # Plot the specified point, its limit and its tangent line
        plt.plot(
            point, limit,
            marker='.',
            markerfacecolor='white',
            color='red', markersize=10.0,
            label=r"Point at $(%s, %s)$" % (sp.latex(point), sp.latex(limit)),
            zorder=10
        )

        # TANGENT PLOTTING
        # ----------------

        # try:
        #      x_range_tangent = np.linspace(float(point) - 1, float(point) + 1, 100)
        #      slope = sp.lambdify(sym, sp.diff(function), 'numpy') (point)
        #      tangent_line = slope * (x_range_tangent - point) + sp.lambdify(sym, function, 'numpy')(point)

        #      plt.plot(
        #          x_range_tangent, tangent_line,
        #          linestyle='--', color='red'
        #      )
        # except:
        #     pass


    # Plot the function
    plt.plot(
        x_vals, y_vals,
        label=r"$f(x)=%s$" % sp.latex(function),
        color='blue',
        zorder=3
    )

    plt.xlim(*range_x)
    plt.ylim(*range_y)

    # Annotate the plot with arrows and text if specified
    if annotate:
        dec(ax, point, limit)

    output_dir = 'static/images/grafics'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(os.path.join(output_dir, f"{index}-plot.png"),format='png')
    return 1

def table_limit(function, sym, point):
    """
    Generate an approximation table for a given function around a specified point.

    ## Parameters:
    - function: The function to approximate.
    - sym: The variable of the function.
    - point: The point around which the approximation is done.

    ## Returns:
    - Void: Displays the approximation table using IPython.display.
    """

    desired_point = point

    def create_approximation_points(limit_value, distances: List[float]=[0.1, 0.01, 0.001, 0.0001, 0.00001]) -> Tuple:
        """
        Create approximation points around a given limit value.

        ## Parameters:
        - limit_value: The value around which to create the approximation points.
        - distances: Distances from the limit value for each approximation point. Defaults to [0.1, 0.01, 0.001, 0.0001, 0.00001].

        ## Returns:
        - tuple: A tuple containing two lists - left_points and right_points, representing the points to the left and right of the limit value respectively.
        """
        left_points = [limit_value - d for d in distances]
        right_points = [limit_value + d for d in distances]
        return left_points, right_points

    # Generate approximation points
    left_points, right_points = create_approximation_points(desired_point)

    # Calculate f(x) values for the points on the left and right
    left_values = [function.subs(sym, point).evalf() for point in left_points]
    right_values = [function.subs(sym, point).evalf() for point in right_points]

    # Prepare the output table in LaTeX format
    table_latex = "\\begin{array}{|c|c|c|c|}\n"
    table_latex += "\\hline\n"
    table_latex += "\\text{Left Approx.} & \\text{Value} & \\text{Right Approx.} & \\text{Value} \\\\\n"
    table_latex += "\\hline\n"

    for i in range(len(left_points)):
        table_latex += f"{sp.latex(left_points[i])} & {sp.latex(left_values[i])} & {sp.latex(right_points[i])} & {sp.latex(right_values[i])} \\\\\n"

    table_latex += "\\hline\n"
    table_latex += "\\end{array}"

    # Display the table using IPython.display
    steps.append((table_latex))

warnings.filterwarnings("ignore")

def lim(symbol, point):
    return f"\\lim_{{{symbol} \\to {sp.latex(point)}}}"

class Substitution:
    def __init__(self, expression: str, symbol: sp.Symbol, point, debug=False, derive=False):
        self.expression = expression # String expression
        self.expression_sp = sp.sympify(expression) # Sympified expression
        self.expression_latex = sp.latex(self.expression_sp) # Latex expression

        self.symbol = symbol
        self.point = point
        self.debug = debug
        self.derive = derive

    def simple_solve(self):
        expression_subs = self.expression_sp.subs(self.symbol, self.point)
        try:
            if not isnan(expression_subs):
                return expression_subs
            else:
                return None
        except Exception as e:
            return None

    def solve(self):
        try:
            print("--- Lim expression ---") if self.debug else ''
            lim_expression = f"{lim(self.symbol, self.point)} {self.expression_latex} \\\\"
            steps.append((lim_expression))
            steps.append('\\\\')

            print("--- Substitution expression ---") if self.debug else ''
            substitution_expression = f"{lim(self.symbol, self.point)} {self.expression_latex.replace(str(self.symbol), f'({sp.latex(self.point)})')} \\\\"
            steps.append((substitution_expression)) if substitution_expression != lim_expression else ''
            steps.append('\\\\') if substitution_expression != lim_expression else ''

            if not self.expression_sp.is_rational_function:  # If expression is not a fraction
                print("--- Evaluated expression ---") if self.debug else ''
                evaluated_expression = f"{lim(self.symbol, self.point)} {sp.latex(self.expression_sp.subs(self.symbol, self.point))} \\\\"
                steps.append((evaluated_expression)) if evaluated_expression != substitution_expression else ''
                steps.append('\\\\') if evaluated_expression != substitution_expression else ''

                return eval(self.expression)

            numer = self.expression_sp.as_numer_denom()[0]  # Numerator
            denom = self.expression_sp.as_numer_denom()[1]  # Denominator

            numer_subs = numer.subs(self.symbol, self.point)
            denom_subs = denom.subs(self.symbol, self.point)

            if str(denom_subs) == 'nan': ## SCUFFED
                denom_subs = oo

            print("--- Fraction expression ---") if self.debug else ''
            if denom_subs != 1:
                fraction_expression = f"{lim(self.symbol, self.point)} \\frac{{{sp.latex(numer_subs)}}}{{{sp.latex(denom_subs)}}} \\\\"
            else:
                fraction_expression = f"{lim(self.symbol, self.point)} {sp.latex(numer_subs/denom_subs)} \\\\"
            steps.append((fraction_expression)) if fraction_expression != substitution_expression else ''
            steps.append('\\\\') if fraction_expression != substitution_expression else ''

            inf = [sp.oo, -sp.oo, sp.zoo]

            if denom_subs == 0 or (numer_subs in inf and denom_subs in inf):  # Division by zero or infinity
                raise ZeroDivisionError("Indeterminate form")

            else:
                print("--- Final Fraction expression ---") if self.debug else ''
                if numer_subs not in inf and denom_subs in inf:
                    steps.append((fraction_expression + f'= {numer_subs/denom_subs}'))
                steps.append('\\\\')
                return numer_subs / denom_subs

        except Exception as e:
            error_message = f"\\text{{Substitution failed: {e}}} \\\\"
            steps.append((error_message))
            steps.append('\\\\')

            if self.debug:
                print("+", f"Caught Runtime Exception: {e}")

            return None

class Factoring:
    '''
    A factoring algorithm is taken to be the starting point,
    and will be iterated through until it finds either the
    solution where the rational function's denominator is
    no longer an indeterminate form, by substituting it
    with the Substitution class, or where it exhausts
    its options.

    The program stops when all algorithms have been tested
    in all possible different sequences. If we have found
    a way to remove the indeterminate form, then the program
    stops and returns the function. Else, we quit.

    Example (Debug Mode):

    --- Factoring ---
    --- Iteration 0 ---
    + [expression] <-- Factor
    + [expression] <-- Expand
    + [expression] <-- Simplify

    --- Iteration 1 ---
    + [expression] <-- Expand
    + [expression] <-- Simplify
    + [expression] <-- Factor

    --- Iteration 2 ----
    + [expression] <-- Simplify
    + [expression] <-- Factor
    --- Found solution at Iteration 2 ---

    Factoring algorithms can be added by simply adding
    their implementions as methods in the class.

    '''
    def __init__(self, expression, symbol, point, debug=False, derive=False):
        self.symbol = symbol
        self.expression = expression
        self.point = point
        self.debug = debug

        self.derive = derive
        derivative_methods = ["Apply_the_Lhopital_rule_to"] if not derive else [] # Add here the derivative methods you would like to exclude if specified
        self.factoring_algorithms = [getattr(self, func) for func in dir(self) if callable(getattr(self, func)) and not func.startswith("__") and func != "solve" and func not in derivative_methods]

        self.trigUsed = []

    def Factor(self, expression, view=False):
        if not expression.is_rational_function:
            return sp.expand(expression)

        numer, denom = expression.as_numer_denom()

        numer = sp.factor(numer)
        denom = sp.factor(denom)

        if view:
            steps.append((
                f"{lim(self.symbol,self.point)} \\frac{{{sp.latex(eval(str(numer)))}}}{{{sp.latex(eval(str(denom)))}}}"
                ))

        return numer/denom

    def Expand(self, expression, view=False):
        if not expression.is_rational_function:
            return sp.expand(expression)

        numer, denom = expression.as_numer_denom()

        numer = sp.expand(numer)
        denom = sp.expand(denom)

        if view:
            steps.append((
                f"{lim(self.symbol,self.point)} \\frac{{{sp.latex(eval(str(numer)))}}}{{{sp.latex(eval(str(denom)))}}}"
                ))

        return numer/denom

    def Simplify(self, expression, view=False):
        if not expression.is_rational_function:
            return sp.expand(expression)

        numer, denom = expression.as_numer_denom()

        numer = sp.simplify(numer)
        denom = sp.simplify(denom)

        if view:
            steps.append((
                f"{lim(self.symbol,self.point)} \\frac{{{sp.latex(eval(str(numer)))}}}{{{sp.latex(eval(str(denom)))}}}"
                ))

        return numer/denom

    def Rationalize_denominator_of(self, expression, view=False):
        if not expression.is_rational_function or not expression.has(sp.Pow):
            return expression

        numer, denom = expression.as_numer_denom()
        if denom.has(sp.Pow):
            conjugate_denom = (denom.args[0] - denom.args[1])
            rationalized_expr = sp.Mul(
                sp.Mul(numer, conjugate_denom, evaluate=False), # Numerator * Conjugate
                sp.Pow(sp.Mul(denom, conjugate_denom, evaluate=False), -1, evaluate=False), # ( Denominator * Conjugate ) ^ -1
                evaluate=False)

            if view:
                numer_latex = sp.latex(eval( str(numer) ))
                denom_latex = sp.latex(eval( str(denom) ))
                conjugate_denom_latex = sp.latex(eval( str(conjugate_denom) ))

                rationalized_numer_latex = sp.latex(sp.Mul(numer, conjugate_denom, evaluate=False))
                rationalized_denom_latex = sp.latex(sp.Mul(denom, conjugate_denom, evaluate=False))

                steps.append((
                    f"{lim(self.symbol,self.point)} \\frac{{{numer_latex}}}{{{denom_latex}}}" +
                    "\\times" +
                    f"\\frac{{{conjugate_denom_latex}}}{{{conjugate_denom_latex}}}"
                    ))
                steps.append('\\\\')
                steps.append((
                    f"{lim(self.symbol,self.point)} \\frac{{{rationalized_numer_latex}}}{{{rationalized_denom_latex}}}"
                    ))
            return rationalized_expr
        else:
            return expression

    def Rationalize_numerator_of(self, expression, view=False):
        if not expression.is_rational_function or not expression.has(sp.Pow):
            return expression

        numer, denom = expression.as_numer_denom()
        if numer.has(sp.Pow):
            conjugate_numer = (numer.args[0] - numer.args[1])
            rationalized_expr = sp.Mul(
                sp.Mul(numer, conjugate_numer, evaluate=False), # Numerator * Conjugate
                sp.Pow(sp.Mul(denom, conjugate_numer, evaluate=False), -1, evaluate=False), # ( Denominator * Conjugate ) ^ -1
                evaluate=False)

            if view:
                numer_latex = sp.latex(eval( str(numer) ))
                denom_latex = sp.latex(eval( str(denom) ))
                conjugate_numer_latex = sp.latex(eval( str(conjugate_numer) ))

                rationalized_numer_latex = sp.latex(sp.Mul(numer, conjugate_numer, evaluate=False))
                rationalized_denom_latex = sp.latex(sp.Mul(denom, conjugate_numer, evaluate=False))

                steps.append((
                    f"{lim(self.symbol,self.point)} \\frac{{{numer_latex}}}{{{denom_latex}}}" +
                    "\\times" +
                    f"\\frac{{{conjugate_numer_latex}}}{{{conjugate_numer_latex}}}"
                    ))
                steps.append('\\\\')
                steps.append((
                    f"{lim(self.symbol,self.point)} \\frac{{{rationalized_numer_latex}}}{{{rationalized_denom_latex}}}"
                    ))
            return rationalized_expr
        else:
            return expression

    def Apply_trigonometric_identities_to(self, expression, view=False):
        trig_identities = {
            'sin(2*x)': 2 * sp.sin(self.symbol) * sp.cos(self.symbol),
            'cos(2*x)': sp.cos(self.symbol)**2 - sp.sin(self.symbol)**2,
            'tan(2*x)': 2 * sp.tan(self.symbol) / (1 - sp.tan(self.symbol)**2),
            '1 - cos(x)**2': sp.sin(self.symbol)**2,
            'sin(x)**2': (1 - sp.cos(2*self.symbol)) / 2,
            'cos(x)**2': (1 + sp.cos(2*self.symbol)) / 2,
            'sin(x)*cos(x)': sp.sin(2*self.symbol) / 2,
            'tan(x)': sp.sin(self.symbol) / sp.cos(self.symbol),
            'cot(x)': sp.cos(self.symbol) / sp.sin(self.symbol),
            'sec(x)': 1 / sp.cos(self.symbol),
            'csc(x)': 1 / sp.sin(self.symbol)
        }

        contains_trig = expression.has(sp.sin) or expression.has(sp.cos) or expression.has(sp.tan)

        if sp.limit(sp.sympify(expression),self.symbol,self.point) not in [sp.oo, -sp.oo]:
            trig_identities.update({
                'sin(x)': sp.series(sp.sin(self.symbol), self.symbol, 0, 6).removeO(),
                'cos(x)': sp.series(sp.cos(self.symbol), self.symbol, 0, 6).removeO()
            })

        self.trigUsed.pop() if self.trigUsed and view else ''

        if contains_trig:
            identity_used = None

            for identity, replacement in trig_identities.items():
                if expression.has(sp.sympify(identity)) and not (identity in self.trigUsed):
                    with sp.evaluate(False):
                        expression = expression.subs(sp.sympify(identity), replacement)
                    identity_used = identity, replacement
                    self.trigUsed.append(identity)
                    break

            if view:
                steps.append((
                    f"\\text{{The function is valid for the trigonometric identity:}} {sp.latex(sp.sympify(identity_used[0]))} = {sp.latex(identity_used[1])}"
                )) if identity_used is not None else ''

                steps.append('\\\\')

                steps.append((
                    f"{lim(self.symbol, self.point)} {sp.latex(expression)}"
                    ))

            print(f"+ ↳ Trigonometric rule used {identity_used[0]} = {identity_used[1]}") if self.debug and identity_used is not None else ''

        return expression

    def Apply_the_Lhopital_rule_to(self, expression, view=False):
        if not expression.is_rational_function:
            return expression

        numer = expression.as_numer_denom()[0]
        denom = expression.as_numer_denom()[1]

        numer_diff = sp.diff(numer)
        denom_diff = sp.diff(denom)

        with sp.evaluate(False):
            derivative_expr = numer_diff / denom_diff

        if self.debug:
            print(f"+ Numerator derivative: d/dx{numer_diff}")
            print(f"+ Denominator derivative: d/dx{denom_diff}")

        if view:
            steps.append((
                f"{lim(self.symbol, self.point)} \\frac{{\\frac{{d}}{{d{self.symbol}}} ({sp.latex(numer)})}}{{\\frac{{d}}{{d{self.symbol}}} ({sp.latex(denom)})}}"
                ))
            steps.append('\\\\')
            steps.append((
                f"{lim(self.symbol, self.point)} {sp.latex(derivative_expr)}"
                ))

        return derivative_expr

    def Factor_by_strongest_term_of(self, expression, view=False):
        if not expression.is_rational_function:
            return expression

        numer, denom = expression.as_numer_denom()

        with sp.evaluate(True):
            if numer.is_polynomial(self.symbol) and denom.is_polynomial(self.symbol):
                numer = sp.simplify(numer)
                denom = sp.simplify(denom)
                numer_degree = sp.degree(numer, self.symbol)
                denom_degree = sp.degree(denom, self.symbol)

                if numer_degree == denom_degree:

                    numer_coeff = numer.coeff(self.symbol, numer_degree)
                    denom_coeff = denom.coeff(self.symbol, denom_degree)

                    result = numer_coeff / denom_coeff

                    if view:
                        steps.append((
                            f"{lim(self.symbol, self.point)} \\frac{{{sp.latex(numer)}}}{{{sp.latex(denom)}}} = \\frac{{{sp.latex(numer_coeff)}}}{{{sp.latex(denom_coeff)}}} = {sp.latex(result)}"
                        ))

                    return result

            return expression

    def solve(self):
        for i in range(len(self.factoring_algorithms)):
            reordered_algorithms = self.factoring_algorithms[i:] + self.factoring_algorithms[:i]
            order = []
            solution = self.expression

            print("+ Iteration {:d}".format(i)) if self.debug else ''

            for algorithm in reordered_algorithms:
                factored_expr = algorithm(sp.sympify(solution))

                if str(factored_expr).replace(" ","").strip() != str(solution).replace(" ","").strip():
                    order.append((algorithm,solution))

                subs = Substitution(str(factored_expr), self.symbol, self.point, self.debug)
                subs_r = subs.simple_solve()

                print("+", factored_expr, " <-- ", algorithm.__name__) if self.debug else ''

                if subs_r is not None:

                    steps.append((
                        f"{lim(self.symbol,self.point)}{sp.latex(eval(self.expression))}"
                        ))
                    steps.append('\\\\')

                    for ordered_algorithm in order:
                        print(f"--- {ordered_algorithm[0].__name__.replace('_', ' ')} ---") if self.debug else ''

                        steps.append((
                            f"\\text{{{ordered_algorithm[0].__name__.replace('_', ' ')} the function}}"
                            ))
                        steps.append('\\\\')

                        ordered_algorithm[0](sp.sympify(ordered_algorithm[1]),view=True)
                        steps.append('\\\\')

                    subs.solve()

                    return subs_r

                else:
                    solution = factored_expr
        return None

class TableApproximation:
    def __init__(self, expression, symbol, point, debug=False, derive=False):
        self.symbol = symbol
        self.expression = expression
        self.point = point
        self.debug = debug
        self.derive = derive

    def solve(self):
        steps.append((
            f"\\text{{Laterally approach the function to}} \\, {sp.latex(self.point)}"
            ))
        steps.append('\\\\')
        table_limit(sp.sympify(self.expression), self.symbol, self.point)
        steps.append('\\\\')
        steps.append('\\\\')
        steps.append('\\\\')
        steps.append('\\\\')
        lim_left = sp.limit(self.expression,self.symbol,self.point, dir='-')
        lim_right = sp.limit(self.expression,self.symbol,self.point, dir='+')

        steps.append((
            f'\\lim_{{{x} \\to {self.point}-}}{sp.latex(eval(str(self.expression)))}\\to {sp.latex(sp.sympify(str(lim_left)))}'
            ))
        steps.append('\\\\')
        steps.append((
            f'\\lim_{{{x} \\to {self.point}+}}{sp.latex(eval(str(self.expression)))}\\to {sp.latex(sp.sympify(str(lim_right)))}'
            ))
        steps.append('\\\\')
        if lim_left != lim_right:
            return lim_left,lim_right
        return lim_left

class StepByStepLimitSolver:
    def __init__(self, expression, symbol, point, debug=False, derive=False):
        self.symbol = symbol
        self.expression = expression
        self.point = point
        self.debug = debug
        self.derive = derive

        self.methods = [Substitution, Factoring, TableApproximation]

        print("--- DEBUG MODE ---") if debug else ''

    def solve(self):
        for method_class in self.methods:
            print(f"--- {method_class.__name__} ---") if self.debug else ''
            method = method_class(self.expression, self.symbol, self.point, self.debug, self.derive)
            result = method.solve()

            if result is not None:
                try:
                    steps.append((f"{lim(self.symbol,self.point)}{sp.latex(eval(str(self.expression)))} = {sp.latex(sp.sympify(str(result)))}"))
                except Exception as e:
                    if self.debug:
                        print(e, end="\n\n")
                        print(eval(result))
                    steps.append((f"{lim(self.symbol,self.point)}{sp.latex(eval(str(self.expression)))} = {sp.latex(eval(result))}"))
                steps.append('\\\\')
                # plot_limit(sp.sympify(self.expression),self.symbol,self.point)
                return result
        print("Unable to find a solution.")

def read_input(input: str):
    """Extract e, z, z0 from input."""
    _e_temp, _z_temp, _z0_temp = '', '', ''

    try:
        _substrings = [
            input.index('lim of'),
            input.index('lim of')+len('lim of'),

            input.index('as'),
            input.index('as')+len('as'),

            input.index('tends to'),
            input.index('tends to')+len('tends to')
        ]
        _e_temp += input[_substrings[1]:_substrings[2]].strip()
        _z_temp += input[_substrings[3]:_substrings[4]].strip()
        _z0_temp += input[_substrings[5]:].strip()

    except:
        return "Incorrect syntax, no limit found! Correct use: `lim of (expression) as (symbol) tends to (point)`"

    return _e_temp, _z_temp, _z0_temp

def solve_limit(input: str, debug: bool=False, derive: bool=False):
    '''
    Function implementation of the StepByStepLimitSolver.

    Parameters:
    - input: str -> text input that will be converted into a standard expression
    - debug: bool -> boolean indicating whether or not to initiate in debug mode
    - derive: bool -> boolean indicating whether or not to include derivative solutions, such as Lhopital

    Usage:
    > solve_limit("lim of [expression] as [symbol] tends to [point]")

    Example:
    >>> import sympy as sp
    >
    >>>> x = sp.symbols('x')
    > x
    >>> solve_limit('lim of (x**2 - 1)/(x-1) as x tends to 1')
    > --- Substitution ---
    > (x**2-1)/(x-1)
    > ((1)**2-1)/((1)-1)
    > 0/0
    > Substitution failed: division by zero
    >
    > --- Factoring ---
    > (x**2-1)/x
    > --- Factor ---
    > (x**2-1)/(x-1)
    > ((x-1)*(x+1))/(x-1)
    > x+1
    > --- Substitution ---
    > x+1
    > (1)+1
    > 2
    > Result: 2F

    '''
    global steps
    steps = []
    solved = read_input(input)
    if isinstance(solved, tuple):
        expr, sym, point = read_input(input)
    else:
        error = solved
        return error
    solver = StepByStepLimitSolver(expr,sym,point,debug,derive)
    result = solver.solve()
    if result is not None:
        return steps, sp.sympify(expr), sp.sympify(str(sym)), sp.sympify(point)

