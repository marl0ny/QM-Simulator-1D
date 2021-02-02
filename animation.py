"""
Matplotlib animation for graphing the single particle
wavefunction. This module is intended to be independent
of the GUI backend chosen (such as Tkinter), and can
be used without it, by using Matplotlib in interactive
mode and typing command line arguments. 
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from functions import rect, convert_to_function, Function
from qm.constants import Constants
from qm import Wavefunction1D, UnitaryOperator1D
from time import perf_counter
import matplotlib


np.seterr(all='raise')
# Make numpy raise errors instead of warnings.


def scale(x: np.ndarray, scale_val: float) -> np.ndarray:
    """
    Scale x back into a boundary if it exceeds it.

    >>> scale(np.array([0,1,2,3,4,5]), 3)
    array([0. , 0.6, 1.2, 1.8, 2.4, 3. ])

    >>> scale(np.array([-10, 6, 3, 0, 1, 2]), 5)
    array([-5. ,  3. ,  1.5,  0. ,  0.5,  1. ])

    """
    absmaxposval = np.abs(np.amax(x))
    absmaxnegval = np.abs(np.amin(x))
    if (absmaxposval > scale_val or absmaxnegval > scale_val):
        x = scale_val*x/absmaxposval \
            if absmaxposval > absmaxnegval else \
               scale_val*x/absmaxnegval
    return x


def ordinate(number_string: str) -> str:
    """
    Turn numbers of the form '1' into '1st',
    '2' into '2nd', and so on.
    """
    if (len(number_string) >= 2) \
         and (number_string[-2:] == "11"):
        return number_string + "th"
    elif (len(number_string) >= 2) \
         and (number_string[-2:] == "12"):
        return number_string + "th"
    elif (len(number_string) >= 2) \
         and (number_string[-2:] == "13"):
        return number_string + "th"
    elif number_string[-1] == "1":
        return number_string + "st"
    elif number_string[-1] == "2":
        return number_string + "nd"
    elif number_string[-1] == "3":
        return number_string + "rd"
    else:
        return number_string + "th"


def rescale_array(x_prime: np.ndarray, 
                  x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Given an array x that maps to an array y, and array x
    that is transformed to x_prime, apply this same transform
    to y.
    """
    y_prime = np.zeros([len(x)])
    contains_value = np.zeros([len(x)], np.int32)
    for i in range(len(x)):
        index = 0
        min_val = abs(x[i] - x_prime[0])
        for j in range(1, len(x_prime)):
            if abs(x[i] - x_prime[j]) < min_val:
                index = j
                min_val = abs(x[i] - x_prime[j])
        if min_val < (x[1] - x[0]):
            if contains_value[index] == 0:
                y_prime[index] = y[i]
                contains_value[index] = 1
            else:
                contains_value[index] += 1
                y_prime[index] = (y[i]/contains_value[index]
                                  + y_prime[index]*(
                                  contains_value[index] - 1.0)/
                                  contains_value[index])
    i = 0
    while i < len(y_prime):
        if (i + 1 < len(y_prime) 
            and contains_value[i+1] == 0
            ):
            j = i + 1
            while (contains_value[j] == 0 
                   and j < len(y_prime) - 1
                   ):
                j += 1
            for k in range(i+1, j):
                y_prime[k] = y_prime[i] + ((k - i)/(j - i))*(
                             y_prime[j] - y_prime[i])
            i = j - 1
        i += 1
    return y_prime


class QuantumAnimation(Constants):
    """
    Class for QM Animation

    Attributes:
    fpi [int]: number of time evolutions per animation frame
    psi [Wavefunction1D]: wavefunction
    U_t [UnitaryOperator1D]: The time evolution operator
    x [np.ndarray]: position as an array
    V_x [np.ndarray]: potential as an array
    psi_name [str]: String name of the wavefunction
    psi_latex [str]: LaTEX name of the wavefunction
    V_name [str]: String name of the potential
    V_latex [str]: LaTEX name of the potential
    figure [plt.figure.Figure]:
            matplotlib figure object
    ax [plt.axes._subplots.AxesSubplot]:
        matplotlib ax object
    lines [list[plt.text.Text, plt.lines.Line2D]]:
          List of matplotlib artist objects
    main_animation [animation.FuncAnimation]:
                    Main animation
    """

    def __init__(self, function="np.exp(-0.5*((x-0.25)/0.05)**2)",
                 potential="(x)**2/2"):
        """
         Initialize the animation.
        """

        super().__init__()

        # String Attributes
        self._KE_ltx = r"-\frac{\hbar^2}{2m} \frac{d^2}{dx^2}"
        self._lmts_str = r"  %s$ \leq x \leq $%s" % (str(np.round(self.x0, 1)),
                                                     str(np.round(self.L +
                                                                  self.x0, 1)))
        self._msg = ""  # Temporary messages in the text
        # box in the upper left corner
        self._main_msg = ""  # Primary messages in this same text box.
        self._main_msg_store = "" # Store the primary message
        self.psi_name = ""  # Name of the wavefunction
        self.psi_latex = ""  # LaTEX name of the wavefunction
        self.V_name = ""   # Name of the potential
        self.V_latex = ""  # LaTEX name of the potential
        self.identity_matrix = np.identity(self.N, np.complex128)

        # Ticking int attributes
        self.fpi = 1    # Set the number of time evolutions per animation frame
        self._t = 0     # Time that has passed
        self._msg_i = 0  # Message counter for displaying temporary messages
        self.fps = 30    # frames per second
        self.fps_total = 0  # Total number of fps
        self.avg_fps = 0  # Average fps
        self.ticks = 0    # total number of ticks

        # The x-axis x ticks
        self._x_ticks = []

        self.t_perf = [1.0, 0.]

        # Set the dpi (Resolution in plt.figure())
        self._dpi = 120

        # Boolean Attributes
        # Display the probability function or not
        self._display_probs = False

        # Scale y value
        self._scale_y = 1.0

        # Whether to show momentum p or to show position x
        self._show_p = False

        # Whether to show energy level or not.
        self._show_energy_levels = False

        # Whether to show expectation value or not
        self._show_exp_val = False

        # tuple containing the position of the message
        self._msg_pos = (0, 0)

        # Numpy array of positions
        self.x = np.linspace(self.x0,
                             (self.L + self.x0),
                             self.N)

        # the parameters
        self.psi_base = None
        self.psi_params = {}
        self.V_base = None
        self.V_params = {}

        Function.add_function("arg", lambda theta: np.exp(2.0j*np.pi*theta))
        Function.add_function("ees", lambda n, x:
                               self.get_energy_eigenstate(int(n)) 
                               if np.array_equal(self.x, x) else
                               rescale_array(x, self.x, 
                               np.real(self.get_energy_eigenstate(int(n))))
                               )

        self.set_wavefunction(function)

        self.V_x = None
        self.set_unitary(potential)

        self._init_plots()

    def set_wavefunction(self, psi, normalize=True):
        """Parse input to set the wavefunction attributes.
        """
        if isinstance(psi, str):
            try:
                if psi.strip().replace(".", "").replace("-", "").replace(
                        "e", "").isnumeric():
                    psi_x = float(psi)*np.ones([self.N])
                    self.psi_name = psi
                    self.psi_latex = "$%s$" % psi
                    self.psi = Wavefunction1D(psi_x)
                    self._msg = "$\psi(x, 0) =$ %s" % self.psi_latex
                    self._msg_i = 45
                    if normalize:
                        self.psi.normalize()
                    self.psi_base = None
                    self.psi_params = {}
                else:
                    psi = psi.replace("^", "**")
                    f = Function(psi, "x")
                    self.psi_base = f
                    psi_func = lambda x: f(x, *f.get_tupled_default_values())
                    self.psi_name = str(f)
                    self.psi_latex = "$" + f.latex_repr + "$"
                    self.psi = Wavefunction1D(psi_func)
                    self.psi_params = f.get_enumerated_default_values()
                    self._msg = r"$\psi(x, 0) =$ %s" % self.psi_latex
                    self._msg_i = 45
                    if normalize:
                        self.psi.normalize()
            except (TypeError, AttributeError,
                    SyntaxError, ValueError, NameError) as E:
                print(E)
        elif isinstance(psi, np.ndarray):
            # self.psi_base = None
            # self.psi_params = {}
            self.psi = Wavefunction1D(psi)
            self.psi_name = "wavefunction"
            self.psi_latex = "$\psi(x)$"
            if normalize:
                self.psi.normalize()
        else:
            print("Unable to parse input")

    def set_unitary(self, V):
        """Parse input and set the unitary operator attributes.
        This also sets up the potential function
        attributes in the process.
        """
        if isinstance(V, str):
            try:
                if V.strip().replace(".", "").replace(
                        "-", "").replace("e", "").isnumeric():
                    self.V_name = ""
                    self.V_latex = str(np.round(float(V), 2))
                    if float(V) == 0:
                        V = 1e-30
                        V_f = float(V)*np.ones([self.N])
                        self.U_t = UnitaryOperator1D(np.copy(V_f))
                        self.V_x = 0.0*V_f
                    else:
                        V_f = scale(float(V)*np.ones([self.N]), 15)
                        self.V_x = V_f
                        self.U_t = UnitaryOperator1D(np.copy(V_f))
                        self.V_latex = "%sk" % (self.V_latex) if V_f[0] > 0\
                                       else " %sk" % (self.V_latex)
                    self.V_params = {}
                    self.V_base = None
                else:
                    V = V.replace("^", "**")
                    f = Function(V, "x")
                    self.V = lambda x: f(x, *f.get_tupled_default_values())
                    self.V_x = scale(self.V(self.x), 15)
                    self.V_name = str(f)
                    self.V_latex = "$" + f.multiply_latex_string("k") + "$"
                    self.U_t = UnitaryOperator1D(self.V)
                    self.V_base = f
                    self.V_params = f.get_enumerated_default_values()
            except (TypeError, AttributeError,
                    SyntaxError, ValueError, NameError) as E:
                print(E)
        elif isinstance(V, np.ndarray):
            self.V_params = {}
            self.V_base = None
            self.V = None
            self.V_x = scale(V, 15)
            self.V_name = "V(x)"
            self.V_latex = "$V(x)$"
            self.U_t = UnitaryOperator1D(V)
        else:
            print("Unable to parse input")

        if hasattr(self, "lines"):
            self.update_draw_potential()

    def update_draw_potential(self):
        """
        Update the plot of the potential V(x)
        """

        #Update the actual plots
        if np.amax(self.V_x > 0):
            V_max = np.amax(self.V_x[1:-2])
            self.lines[4].set_ydata(self.V_x/
                                    V_max*self.bounds[-1]*0.95)
            V_max *= self._scale
            self.lines[9].set_text("E = %.0f" % (V_max))
            self.lines[10].set_text("E = %.0f" % (-V_max))
        elif np.amax(self.V_x < 0):
            V_max = np.abs(np.amin(self.V_x[1:-2]))
            self.lines[4].set_ydata(self.V_x/
                                    V_max*self.bounds[-1]*0.95)
            V_max *= self._scale
            self.lines[9].set_text("E = %.0f" % (V_max))
            self.lines[10].set_text("E = %.0f" % (-V_max))
        else:
            V_max = self.bounds[-1]*0.95*self._scale
            self.lines[4].set_ydata(self.x*0.0)
            self.lines[9].set_text("E = %.0f" % (V_max))
            self.lines[10].set_text("E = %.0f" % (-V_max))

        # Update the text display
        if (self.V_latex.replace(".", "").isnumeric() and
                (float(self.V_latex) == 0.)):
            self.set_main_message("$H = %s$, \n%s" % (
                    self._KE_ltx, self._lmts_str))
        elif self.V_latex[1] == "-":
            self.set_main_message("$H = %s $%s, \n%s"%(
                self._KE_ltx, self.V_latex, self._lmts_str)
            )
        else:
            self.set_main_message("$H = %s + $%s, \n%s" % (
                    self._KE_ltx, self.V_latex, self._lmts_str))

    def display_probability(self, *args):
        """
        Show only the probability density |\psi(x)|^2
        (or |\psi(p)|^2).
        """
        self._display_probs = True
        self.lines[1].set_linewidth(1.25)
        self.lines[2].set_alpha(0.)
        self.lines[3].set_alpha(0.)
        if self._show_p:
            self.lines[0].set_text("—— $|\psi(p)|^2$")
        else:
            self.lines[0].set_text("—— $|\psi(x)|^2$")
        self.lines[6].set_alpha(0.)
        self.lines[7].set_alpha(0.)

    def display_wavefunction(self, *args):
        """
        Show the wavefunction \psi(x) and hide the
        probability density.
        """
        self._display_probs = False
        self.lines[1].set_linewidth(0.75)
        self.lines[2].set_alpha(1.)
        self.lines[3].set_alpha(1.)
        if self._show_p:
            self.lines[0].set_text("—— $|\psi(p)|$")
        else:
            self.lines[0].set_text("—— $|\psi(x)|$")
        # self.lines[0].set_text("—— |Ψ(x)|")
        self.lines[6].set_alpha(1.)
        self.lines[7].set_alpha(1.)

    def display_momentum(self, *args):
        """
        Show the wavefunction in the momentum basis.
        """
        self._show_p = True
        psi_text0 = self.lines[0].get_text().replace("x", "p")
        psi_text6 = self.lines[6].get_text().replace("x", "p")
        psi_text7 = self.lines[7].get_text().replace("x", "p")
        self.lines[0].set_text(psi_text0)
        self.lines[6].set_text(psi_text6)
        self.lines[7].set_text(psi_text7)
        # freq = np.fft.fftshift(np.fft.fftfreq(len(self.x), d=self.dx))
        # p = 2*np.pi*freq*self.hbar/self.L
        # for i in range(1, 5):
        #     self.lines[i].set_xdata(p)
        # self.lines[1]
        locs = self.ax.get_xticks()
        labels = self.ax.get_xticklabels()
        self._x_ticks = [text.get_text() for text in labels]
        p_range = (2*np.pi*self.hbar/self.L)*(self.N)/(self.dx*self.N)
        p_ticks = []
        for x in locs:
            if self.N % 2 == 0:
                p0 = -((2*np.pi*self.hbar/self.L)
                       *(self.N/(2*self.dx*self.N)))
            else:
                p0 = -((2*np.pi*self.hbar/self.L)*
                       ((self.N - 1)/(2*self.dx*self.N)))
            # print(x)
            p_tick = p0 + p_range*((float(str(x)) - self.x[0])/
                                   (self.x[-1] - self.x[0]))
            p_tick = np.round(p_tick, 1)
            p_ticks.append(p_tick)
        # xp_dict = {self.x[i]: p[i] for i in 
        #            [self.N//10, 3*self.N//10, 5*self.N//10,
        #             7*self.N//10, 9*self.N//10]}
        # p_ticks = [xp_dict[key] for key in xp_dict]
        self.lines[4].set_alpha(0.0)
        self._main_msg_store = self._main_msg
        self._main_msg = ""
        self.lines[5].set_text(self._main_msg)
        self.lines[8].set_alpha(0.0)
        if not self._show_energy_levels:
            self.lines[9].set_alpha(0.0)
            self.lines[10].set_alpha(0.0)
            self.lines[11].set_alpha(0.0)
        self.toggle_blit()
        self.ax.set_xticklabels(p_ticks)
        self.ax.set_xlabel("p")
        # self.ax.set_xlim(np.amin(p), np.amax(p))
        self.toggle_blit()

    def display_position(self, *args):
        """
        Show the wavefunction in the position basis.
        """
        self._show_p = False
        psi_text0 = self.lines[0].get_text().replace("(p)", "(x)")
        psi_text6 = self.lines[6].get_text().replace("(p)", "(x)")
        psi_text7 = self.lines[7].get_text().replace("(p)", "(x)")
        self.lines[0].set_text(psi_text0)
        self.lines[6].set_text(psi_text6)
        self.lines[7].set_text(psi_text7)
        self.lines[4].set_alpha(1.0)
        self._main_msg = self._main_msg_store
        self.lines[5].set_text(self._main_msg)
        self.lines[8].set_alpha(1.0)
        self.lines[9].set_alpha(1.0)
        self.lines[10].set_alpha(1.0)
        self.lines[11].set_alpha(1.0)
        self.toggle_blit()
        self.ax.set_xticklabels(self._x_ticks)
        self.ax.set_xlabel("x")
        self.ax.set_xlim(self.x[0] - 0.02*(self.x[-1] - self.x[0]),
                         self.x[-1] + 0.02*(self.x[-1] - self.x[0]))
        self.toggle_blit()

    def measure_energy(self, *args):
        """
        Measure the energy. This collapses the wavefunction
        to the most probable energy eigenstate.
        """
        if not hasattr(self.U_t, "energy_eigenvalues"):
            self.U_t.set_energy_eigenstates()
        EE = np.sort(np.real(self.U_t.energy_eigenvalues))
        EEd = {E: (i + 1) for i, E in enumerate(EE)}
        E = self.psi.set_to_eigenstate(
            self.U_t.energy_eigenvalues,
            self.U_t.energy_eigenstates)
        n = ordinate(str(EEd[np.real(E)]))
        self._msg = "Energy E = %s\n(%s energy level)" % (
            str(np.round(np.real(E), 1)), n)
        self._msg_i = 50
        self.update_expected_energy_level()
        
    def _set_eigenstates(self) -> None:
        """
        Helper functions for lower and higher energy eigenstate.
        """
        # TODO: In this function, a new attribute (U_t._nE, where U_t
        # is of type UnitaryOperator1D) is defined outside
        # of the original function. Don't do this.
        if not hasattr(self.U_t, "energy_eigenvalues"):
            self.U_t.set_energy_eigenstates()
        if not hasattr(self.U_t, "_nE"):
            self.U_t._nE = 0
            self._nE = 0
            ind = np.argsort(
                np.real(self.U_t.energy_eigenvalues))
            eigvects = np.copy(self.U_t.energy_eigenstates).T
            eigvals = np.copy(self.U_t.energy_eigenvalues)
            for i, j in enumerate(ind):
                eigvals[i] = self.U_t.energy_eigenvalues[j]
                eigvects[i] = self.U_t.energy_eigenstates.T[j]
            self.U_t.energy_eigenvalues = eigvals
            self.U_t.energy_eigenstates = eigvects.T
            
    def set_to_eigenstate(self, energy: float, scale_y: float = 1.0) -> None:
        """
        Given an energy eigenvalue, set the wavefunction
        to the corresponding energy eigenstate. Note that
        it is assumed that the eigenstates are sorted in conjunction
        with their energies from lowest to highest.
        """
        energy_range = 8*scale_y*self.U_t._scale
        # energy_range = self.U_t.energy_eigenvalues[-1] - self.U_t.energy_eigenvalues[0]
        for n, eigval in enumerate(self.U_t.energy_eigenvalues):
            if np.abs(eigval - energy) < energy_range/100:
                self.psi.x = self.U_t.energy_eigenstates.T[n]
                self.psi.normalize()
                n = ordinate(str(n + 1))
                self._msg = "Energy E = %s\n(%s energy level)" % (
                    str(np.round(np.real(eigval), 1)), n)
                self._msg_i = 50
                return

    def lower_energy_eigenstate(self, *args) -> None:
        """
        Go to a lower energy eigenstate
        """
        self._set_eigenstates()
        self.U_t._nE -= 1 if self.U_t._nE > 0 else 0
        n = self.U_t._nE
        E = np.real(self.U_t.energy_eigenvalues[n])
        self.psi.x = self.U_t.energy_eigenstates.T[n]
        self.psi.normalize()
        n = ordinate(str(n + 1))
        self._msg = "Energy E = %s\n(%s energy level)" % (
            str(np.round(np.real(E), 1)), n)
        self._msg_i = 50
        self.update_expected_energy_level()

    def get_energy_eigenstate(self, n) -> None:
        """
        Get an eigenstate, given the energy level.
        """
        n -= 1
        self._set_eigenstates()
        if n < 0:
            raise IndexError("energy level enumeration starts from 1.")
        if n >= self.N:
            raise IndexError
        psi = np.copy(self.U_t.energy_eigenstates.T[n])
        return psi

    def higher_energy_eigenstate(self, *args) -> None:
        """
        Go to a higher energy eigenstate
        """
        self._set_eigenstates()
        n_eigvals = len(self.U_t.energy_eigenvalues)
        self.U_t._nE += 1 if self.U_t._nE < n_eigvals - 1 else 0
        n = self.U_t._nE
        E = np.real(self.U_t.energy_eigenvalues[n])
        self.psi.x = self.U_t.energy_eigenstates.T[n]
        self.psi.normalize()
        n = ordinate(str(n + 1))
        self._msg = "Energy E = %s\n(%s energy level)" % (
            str(np.round(np.real(E), 1)), n)
        self._msg_i = 50
        self.update_expected_energy_level()

    def measure_position(self, *args):
        """
        Measure the position. This collapses the wavefunction
        to the most probable position eigenstate.
        """
        x = self.psi.set_to_eigenstate(self.x, self.identity_matrix, smear=True)
        self._msg = "Position x = %s" % (str(np.round(x, 3)))
        self._msg_i = 50
        self.update_expected_energy_level()

    def measure_momentum(self, *args):
        """
        Measure the momentum. This collapses the wavefunction
        to the most probable momentum eigenstate.
        """
        p = self.psi.set_to_momentum_eigenstate()
        freq = str(int(p*(1/(2*np.pi*self.hbar/self.L))))
        self._msg = "Momentum p = %s\n(k = %s)" % (
            str(np.round(p, 3)), freq)
        self._msg_i = 50
        self.update_expected_energy_level()

    def set_m(self, m, *args):
        """
        Change the mass of the particle
        """
        self.m = m
        self.psi.m = m
        self.U_t.m = m
        self.set_unitary(self.V_x)

    def _change_constants(self, hbar, *args):
        """
        Change constants
        """
        self.hbar = hbar
        self.psi.hbar = hbar
        self.U_t.hbar = hbar
        self.set_unitary(self.V_x)

    def set_main_message(self, message: str) -> None:
        """
        Set the main message, i.e. the text at the top left
        of the plot.
        """
        if self._show_p:
            self._main_msg_store = message
        else:
            self.lines[5].set_text(message)
            self._main_msg = message

    def set_scale_y(self) -> float:
        """
        Set the scale y value.
        The scale y value determines how potential values shown
        on the plot is scaled to its actual values.
        """
        # TODO Refactor everything in here!
        if not self.potential_is_reshaped:
            if np.amax(self.V_x > 0):
                self._scale_y = np.amax(self.V_x[1:-2])/(
                    self.bounds[-1]*0.95)
            elif np.amax(self.V_x < 0):
                self._scale_y = np.abs(np.amin(self.V_x[1:-2]))/(
                    self.bounds[-1]*0.95)
            else:
                self._scale_y = 1.0
        else:
            self._scale_y = self.scale_y

    def update_expected_energy_level(self) -> None:
        """
        Update the expected energy level.
        """
        if self._show_energy_levels:
            exp_energy = self.psi.expectation_value(
                                           self.U_t.energy_eigenvalues,
                                           self.U_t.energy_eigenstates)
            exp_energy_show = exp_energy/(self._scale_y*self.U_t._scale)
            self.line11.set_ydata([exp_energy_show,
                                   exp_energy_show])
        
    def update_energy_levels(self) -> None:
        """
        Update the graph of the energy levels.
        """
        if self._show_energy_levels:
            if not hasattr(self.U_t, "_nE"):
                self._set_eigenstates()
            self.set_scale_y()
            q = np.array([(self.x[0] if ((i - 1)//2) % 2 == 0
                           else self.x[-1]) for i in
                          range(2*len(self.U_t.energy_eigenvalues) - 1)])
            e = np.array([self.U_t.energy_eigenvalues[i//2]
                          for i in
                          range(2*len(self.U_t.energy_eigenvalues) - 1)])
            e = e/(self._scale_y*self.U_t._scale)
            self.line10.set_xdata(q)
            self.line10.set_ydata(e)
            self.update_expected_energy_level()

    def show_energy_levels(self) -> bool:
        """
        """
        return self._show_energy_levels

    def toggle_energy_levels(self) -> None:
        """
        Toggle whether energy levels are shown or not.
        """
        self.set_scale_y()
        if self._show_p:
            alpha = 0.0 if self.lines[9].get_alpha() == 1.0 else 1.0
            self.lines[9].set_alpha(alpha)
            self.lines[10].set_alpha(alpha)
            self.lines[11].set_alpha(alpha)
        if not self._show_energy_levels:
            if not hasattr(self.U_t, "_nE"):
                self._set_eigenstates()
            energy_range = np.abs(
                self.U_t.energy_eigenvalues[-1] - 
                self.U_t.energy_eigenvalues[0])
            q = np.array([(self.x[0] if ((i - 1)//2)%2 == 0
                           else self.x[-1]) for i in range(
                                   2*len(self.U_t.energy_eigenvalues) - 1)])
            e = np.array([self.U_t.energy_eigenvalues[i//2]
                          for i in range(
                                  2*len(self.U_t.energy_eigenvalues) - 1)])
            e = e/(self._scale_y*self.U_t._scale)
            exp_energy = self.psi.expectation_value(
                                       self.U_t.energy_eigenvalues,
                                       self.U_t.energy_eigenstates)
            exp_energy_show = exp_energy/(self._scale_y*self.U_t._scale)
            line10, = self.ax.plot(q, e,
                                  linewidth=0.25,
                                  animated=True,
                                  color="darkslategray")
            expected_energy_show = exp_energy/(self._scale_y*self.U_t._scale)
            line11, = self.ax.plot([self.x[0], self.x[-1]],
                                   [expected_energy_show,
                                    expected_energy_show],
                                   animated=True,
                                   color="gray")
            self.line10 = line10
            self.line11 = line11
            self.lines.append(self.line10)
            self.lines.append(self.line11)
            self.line10.set_alpha(0.75)
            self.line11.set_alpha(0.75)
        else:
            self.line10.set_alpha(0.0)
            self.line11.set_alpha(0.0)
            self.lines.pop()
            self.lines.pop()
        self._show_energy_levels = not self._show_energy_levels

    def toggle_expectation_values(self) -> None:
        """
        Toggle expectation values.
        """
        if self._show_exp_val:
            self.lines[5].set_text(self._main_msg)
            self.lines[5].set_position(self._msg_pos)
        else:
            self._msg_pos = self.lines[5].get_position()
            x, y = self._msg_pos
            # If not showing any fps stats or stdevs
            # change y to y*0.8
            # if showing stdevs and avgs change to  
            # If also showing fps stats, change to 0.5
            # If showing both fps stats and stdevs change to 0.1
            self.lines[5].set_position((x, y*0.4))

        self._show_exp_val = not self._show_exp_val

    def _init_plots(self):
        """
        Start the animation, in which the required matplotlib objects
        are initialized and the plot boundaries are determined.
        """

        # Please note, if you change attributes L and x0 in the
        # base Constants class, you may also need to change:
        # - The location of text labels

        # Make matplotlib figure object
        self.figure = plt.figure(dpi=self._dpi)

        # Make a subplot object
        self.ax = self.figure.add_subplot(1, 1, 1)

        # Add a grid
        # self.ax.grid(linestyle="--")

        # Set the x limits of the plot
        xmin = self.x[0]
        xmax = self.x[-1]
        xrange = xmax - xmin
        self.ax.set_xlim(self.x[0] - 0.02*xrange,
                         self.x[-1] + 0.02*xrange)
        self.ax.set_xlabel("x")

        # Set the y limits of the plot
        ymax = np.amax(np.abs(self.psi.x))
        # ymin = np.amin(np.abs(psi.x))
        ymin = -ymax
        yrange = ymax - ymin

        self.ax.get_yaxis().set_visible(False)

        self.ax.set_ylim(ymin-0.1*yrange, ymax+0.1*yrange)

        # Set initial plots with ax.plot.
        # They return the line object which controls the appearance
        # of their plots.
        # Note that the number naming of the variables is not in any
        # logical order.
        # This is due to oversight.
        # TODO: Use a better naming system.

        # line0: Text info for |\psi(x)|^2 or |\psi(x)|
        # line1: |\psi(x)| or |\psi(x)|^2
        # line2: Re(\psi(x))
        # line3: Im(\psi(x))
        # line4: V(x)
        # line5: Text info for Hamiltonian
        # line6: Text info for Im(\psi(x))
        # line7: Text info for Re(\psi(x))
        # line8: Text info for the potential V(x)

        line2, = self.ax.plot(self.x, np.real(self.psi.x),
                              "-",
                              # color="blue",
                              animated=True,
                              # label=r"$Re(\psi(x))$",
                              linewidth=0.5)
        line3, = self.ax.plot(self.x, np.imag(self.psi.x),
                              "-",
                              # color="orange",
                              animated=True,
                              # label=r"$Im(\psi(x))$",
                              linewidth=0.5)
        line1, = self.ax.plot(self.x, np.abs(self.psi.x),
                              animated=True,
                              # label=r"$|\psi(x)|$",
                              color="black",
                              linewidth=0.75)

        if np.amax(self.V_x > 0):
            line4, = self.ax.plot(self.x,
                                  (self.V_x/np.amax(self.V_x[1:-2]))*ymax*0.95,
                                  color="darkslategray",
                                  linestyle='-',
                                  linewidth=0.5)
        elif np.amax(self.V_x < 0):
            line4, = self.ax.plot(self.x,
                                  (self.V_x/
                                   np.abs(np.amin(
                                   self.V_x[1:-2]))*0.95*self.bounds[-1]),
                                  color="darkslategray",
                                  linestyle='-',
                                  linewidth=0.5)
        else:
            line4, = self.ax.plot(self.x,
                                  self.x*0.0,
                                  color="darkslategray",
                                  linestyle='-',
                                  linewidth=0.5)

        line5 = self.ax.text((xmax - xmin)*0.01 + xmin,
                             0.95*ymax,
                             "$H = %s + $ %s, \n%s"%(self._KE_ltx,
                                                     self.V_latex,
                                                     self._lmts_str),
                             # animated=True
                             )

        line0 = self.ax.text((xmax-xmin)*0.01 + xmin,
                             ymin + (ymax-ymin)*0.05,
                             # "—— |Ψ(x)|",
                             "—— $|\psi(x)|$",
                             alpha=1.,
                             animated=True,
                             color="black"
                             )
        line6 = self.ax.text((xmax-xmin)*0.01 + xmin,
                             ymin + (ymax-ymin)*0.,
                             "—— $Re(\psi(x))$",
                             #"—— Re(Ψ(x))",
                             alpha=1.,
                             animated=True,
                             color="C0"
                             )
        line7 = self.ax.text((xmax-xmin)*0.01 + xmin,
                             ymin + (ymax-ymin)*(-0.05),
                             "—— $Im(\psi(x))$",
                             #"—— Im(Ψ(x))",
                             alpha=1.,
                             animated=True,
                             color="C1"
                             )
        
        line8 = self.ax.text((xmax-xmin)*0.01 + xmin,
                             ymin + (ymax-ymin)*(0.1),
                             "—— V(x)",
                             alpha=1.,
                             color="darkslategray")

        # Show the infinite square well boundary
        self.ax.plot([self.x0, self.x0], [-10, 10],
                     color="gray", linewidth=0.75)
        self.ax.plot([self.x0+self.L, self.x0+self.L], [-10, 10],
                     color="gray", linewidth=0.75)

        # Record the plot boundaries
        ymin, ymax = self.ax.get_ylim()
        xmin, xmax = self.ax.get_xlim()
        self.bounds = xmin, xmax, ymin, ymax

        # bottom = np.linspace(ymin, ymin, self.N)
        # self.fill = self.ax.fill_between(self.x, bottom,
        #                                 self.V_x/np.amax(self.V_x[1:-2]),
        #                                 color="gray", alpha=0.05)

        # Store each line in a list.
        self.lines = [line0, line1, line2, line3,
                      line4, line5,
                      line6, line7,
                      line8
                      ]

        # Another round of setting up and scaling the line plots ...
        if np.amax(self.V_x > 0):
            V_max = np.amax(self.V_x[1:-2])
            V_scale = self.V_x/V_max*self.bounds[-1]*0.95
            V_max *= self._scale
            self.lines[4].set_ydata(V_scale)
        elif np.amax(self.V_x < 0):
            V_max = np.abs(np.amin(self.V_x[1:-2]))
            V_scale = self.V_x/V_max*self.bounds[-1]*0.95
            V_max *= self._scale
            self.lines[4].set_ydata(V_scale)
        else:
            self.lines[4].set_ydata(self.x*0.0)

        # Manually plot gird lines
        maxp = self.bounds[-1]*0.95
        self.ax.plot([self.x0, self.x0+self.L], [0., 0.],
                     color="gray", linewidth=0.5, linestyle="--")
        self.ax.plot([self.x0, self.x0+self.L], [maxp, maxp],
                     color="gray", linewidth=0.5, linestyle="--")
        self.ax.plot([self.x0, self.x0+self.L], [-maxp, -maxp],
                     color="gray", linewidth=0.5, linestyle="--")
        # self.ax.plot([0, 0], [-self.bounds[-1], self.bounds[-1]],
        #              color="gray", linewidth=0.5, linestyle = "--")

        # Show where the energy for the potential
        self.lines.append(self.ax.text
                          (xmax*0.7, maxp*0.92,
                           "E = %.0f" % (V_max),
                           color="gray", fontdict={'size':8}))
        self.lines.append(self.ax.text
                          (xmax*0.68, -maxp*0.96,
                           "E = %.0f" % (-V_max),
                           color="gray", fontdict={'size':8}))
        self.lines.append(self.ax.text(xmax*0.8, 0.03, "E = 0",
                          color="gray", fontdict={'size':8}))

        self._main_msg = self.lines[5].get_text()

    def _animate(self, i: int) -> list:
        """Produce a single frame of animation.
        This of course involves advancing the wavefunction
        in time using the unitary operator.
        """

        self.t_perf[0] = self.t_perf[1]
        self.t_perf[1] = perf_counter()

        # Time evolve the wavefunction
        for _ in range(self.fpi):
            self.U_t(self.psi)
            self._t += self.dt

        # Define and set psi depending
        # on whether to show psi in the position
        # or momentum basis.
        if self._show_p:
            psi = self.psi.p
        else:
            psi = self.psi.x

        # Set probability density or absolute value of wavefunction
        if self._display_probs:
            # An underflow error occurs here after
            # measuring the position.
            # Just ignore this for now.
            try:
                self.lines[1].set_ydata(
                    np.real(np.conj(psi)*psi)/3.0)
            except FloatingPointError as E:
                print(E)
        else:
            self.lines[1].set_ydata(np.abs(psi))

        # Set real and imaginary values
        self.lines[2].set_ydata(np.real(psi))
        self.lines[3].set_ydata(np.imag(psi))

        # Find fps stats
        t0, tf = self.t_perf
        self.ticks += 1
        self.fps = int(1/(tf - t0 + 1e-30))
        if self.ticks > 1:
            self.fps_total += self.fps
        self.avg_fps = int(self.fps_total/(self.ticks))
        if self.ticks % 60 == 0:
            pass
            # print_to_terminal("fps: %d, avg fps: %d" % (
            #     self.fps, self.avg_fps))
            # print(self.fps, self.avg_fps)

        # Output temporary text messages
        if self._msg_i > 0:
            self.lines[5].set_text(self._msg)
            self._msg_i += -1
        elif self._msg_i == 0:
            t0, tf = self.t_perf
            self._msg_i += -1
            self.lines[5].set_text(self._main_msg)

        elif self._show_exp_val and self._msg_i < 0:
            if not hasattr(self.U_t, "energy_eigenvalues"):
                self.U_t.set_energy_eigenstates()
            x_mean, x_sigma = \
                self.psi.average_and_standard_deviation(
                    self.x, self.identity_matrix)
            p_mean, p_sigma = \
                self.psi.momentum_average_and_standard_deviation()
            E_mean, E_sigma = \
                self.psi.average_and_standard_deviation(
                self.U_t.energy_eigenvalues,
                self.U_t.energy_eigenstates
                )
            if self.ticks % 5 == 0:
                self.lines[5].set_text(
                    "t = %f\n"
                    # "fps = %i\n"
                    # "avg_fps = %i\n"
                    "<x> = %.2f\n"
                    "<p> = %.2f\n"
                    "<E> = %.0f\n"
                    "σₓ = %.2f\n"
                    "σₚ = %.2f\n"
                    "σᴇ = %.0f" % (
                        self._t,
                        # self.fps,
                        # self.avg_fps,
                        x_mean,
                        p_mean,
                        E_mean,
                        x_sigma,
                        p_sigma,
                        E_sigma
                        )
                    )

        return self.lines

    def animation_loop(self) -> None:
        """Produce all frames of animation.
        """
        self.main_animation = animation.FuncAnimation(
            self.figure, self._animate, blit=True,
            interval=1)

    def get_fpi(self) -> int:
        """Get the number of time evolutions per animation
        frame"""
        return self.fpi

    def set_fpi(self, new_fpi: int) -> None:
        """Set the number of time evolutions per animation frame
        """
        self.fpi = new_fpi

    def toggle_blit(self) -> None:
        """
        Enable or disable blit. This makes it possible to
        update the plot title and axes, which would otherwise
        be static with blit always enabled.
        """
        if self.main_animation._blit:
            base = 100.0
            version_number = 0
            for n in matplotlib.__version__.split('.'):
                version_number += int(n)*base
                base /= 10.0
            if version_number < 330.0:
                self.main_animation._blit_clear(
                    self.main_animation._drawn_artists, 
                    self.main_animation._blit_cache)
            else:
                self.main_animation._blit_clear(
                    self.main_animation._drawn_artists)
            self.main_animation._blit = False
        else:
            # self.main_animation._init_draw()
            self.main_animation._step()
            self.main_animation._blit = True
            self.main_animation._setup_blit()


if __name__ == "__main__":

    from matplotlib import interactive
    interactive(True)
    consts = Constants()
    x = np.linspace(consts.x0, (consts.L + consts.x0), consts.N)
    V = (x)**2/2
    psi = np.cos(3*np.pi*x/consts.L)
    ani = QuantumAnimation(function=psi, potential=V)
    ani.animation_loop()
