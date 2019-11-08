from matplotlib.backends import backend_tkagg
import matplotlib.animation as animation
from sympy import symbols, lambdify, abc, latex
from sympy.parsing.sympy_parser import parse_expr
import numpy as np
import matplotlib.pyplot as plt
from QM_1D_TI import Constants, Wavefunction_1D, Unitary_Operator_1D
from time import perf_counter

np.seterr(all='raise')
# Make numpy raise errors instead of warnings.

def scale(x, scale_val):
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

def rect(x):
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

def delta(x):
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

def coulomb(x):
    """
    1/r potential
    """
    pass
    try:
        dx = (x[-1] - x[0])/len(x)
        return np.array([1e10 if (xi < (0. + dx/2) and xi > (0. - dx/2))
                         else 0. for xi in x])
    except:
        return 1e10 if (x < 0.01 and x > -0.01) \
               else 0.

#Dictionary of modules and user defined functions.
#Used for lambdify from sympy to parse input.
module_list = ["numpy", {"rect":rect}]

def convert_to_function(string, scale_by_k=False):
    """Using the sympy module, parse string input
    into a mathematical expression.
    Returns the original string, the latexified string,
    the mathematical expression in terms of sympy symbols,
    and a lambdified function
    """
    string = string.replace("^","**")
    symbolic_function = parse_expr(string)
    if scale_by_k:
        latexstring = latex(symbolic_function*abc.k)
    else:
        latexstring = latex(symbolic_function)
    lambda_function = lambdify(abc.x, symbolic_function,
                               modules=module_list)
    string = string.replace('*', '')
    latexstring="$" + latexstring + "$"
    return string, latexstring, \
           symbolic_function, lambda_function

def ordinate(number_string):
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

def ordinate_LaTEX(number_string):
    """
    Turn numbers of the form '1' into '1^{st}',
    '2' into '2^{nd}', and so on.
    """
    if (len(number_string) >= 2) \
         and (number_string[-2:] == "11"):
        return number_string + "^{th}"
    elif (len(number_string) >= 2) \
         and (number_string[-2:] == "12"):
        return number_string + "^{th}"
    elif (len(number_string) >= 2) \
         and (number_string[-2:] == "13"):
        return number_string + "^{th}"
    elif number_string[-1] == "1":
        return number_string + "^{st}"
    elif number_string[-1] == "2":
        return number_string + "^{nd}"
    elif number_string[-1] == "3":
        return number_string + "^{rd}"
    else:
        return number_string + "^{th}"

class QM_1D_Animation(Constants):
    """
    Class for QM Animation

    Attributes:
    fpi [int]: number of time evolutions per animation frame
    psi [Wavefunction_1D]: wavefunction
    U_t [Unitary_Operator_1D]: The time evolution operator
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
        self._lmts_str = r"  %s$ \leq x \leq $%s"%(str(np.round(self.x0, 1)),
                                                 str(np.round(self.L + self.x0,
                                                  1)))
        self._msg = "" # Temporary messages in the text
                       # box in the upper left corner
        self._main_msg = ""# Primary messages in this same text box.
        self.psi_name = "" # Name of the wavefunction
        self.psi_latex = ""# LaTEX name of the wavefunction
        self.V_name = ""   # Name of the potential
        self.V_latex = ""  # LaTEX name of the potential


        # Ticking int attributes
        self.fpi = 1    # Set the number of time evolutions per animation frame
        self._t = 0     # Time that has passed
        self._msg_i = 0 # Message counter for displayng temporary messages
        self.fps = 30   # frames per second
        self.fps_total = 0 #Total number of fps
        self.avg_fps = 0  # Average fps
        self.ticks = 0    # total number of ticks

        self.t_perf = [1.0, 0.]

        # Set the dpi (Resolution in plt.figure())
        self._dpi = 120

        # Boolean Attributes
        # Display the probability function or not
        self._display_probs = False

        #Whether to show expectation value or not
        self._show_exp_val = False

        #tuple containing the position of the message
        self._msg_pos = (0, 0)

        # Numpy array of positions
        self.x = np.linspace(self.x0,
                             (self.L + self.x0),
                             self.N)

        self.set_wavefunction(function)
        self.set_unitary(potential)

        self._init_plots()

    def set_wavefunction(self, psi, normalize=True):
        """Parse input to set
        the wavefunction attributes.
        """
        #TODO: -Clean this up
        #      -Add helper functions that are common to
        #       set_wavefunction and set_unitary
        #      -Think of all possible inputs
        #       and sanitization methods

        if isinstance(psi, str):
            try:
                if psi.strip().replace(".","").replace(
                    "-","").replace("e","").isnumeric():
                    psi_x = float(psi)*np.ones([self.N])
                    self.psi_name = psi
                    self.psi_latex = "$%s$"%(psi)
                    self.psi = Wavefunction_1D(psi_x)
                    self._msg = "$\psi(x, 0) =$ %s"%(self.psi_latex)
                    self._msg_i = 45
                    if (normalize):
                        self.psi.normalize()
                else:
                    psi_name, psi_latex, psi_sym , psi_f\
                              = convert_to_function(psi)
                    tmp = np.complex((psi_f(2.)))
                    self.psi_name = psi_name
                    self.psi_latex = psi_latex
                    self.psi = Wavefunction_1D(psi_f)
                    self._msg = "$\psi(x, 0) =$ %s"%(self.psi_latex)
                    self._msg_i = 45
                    if (normalize):
                        self.psi.normalize()
            except (TypeError, AttributeError,
                    SyntaxError, ValueError, NameError) as E:
                print(E)
        elif isinstance(psi, np.ndarray):
            self.psi = Wavefunction_1D(psi)
            self.psi_name = "wavefunction"
            self.psi_latex = "$\psi(x)$"
            if (normalize):
                self.psi.normalize()
        else:
            print("Unable to parse input")

    def set_unitary(self, V):
        """Parse input and
        set the unitary operator attributes.
        This also sets up the potential function
        attributes in the process.
        """
        #TODO: Simplify this code

        if isinstance(V, str):
            try:
                if V.strip().replace(".","").replace(
                    "-","").replace("e","").isnumeric():
                    self.V_name = ""
                    self.V_latex = str(np.round(float(V), 2))
                    if float(V) == 0:
                        V = 1e-30
                        V_f = float(V)*np.ones([self.N])
                        self.U_t = Unitary_Operator_1D(np.copy(V_f))
                        self.V_x = 0.0*V_f
                    else:
                        V_f = scale(float(V)*np.ones([self.N]), 15)
                        self.V_x = V_f
                        self.U_t = Unitary_Operator_1D(np.copy(V_f))
                        self.V_latex = "%sk"%(self.V_latex) if V_f[0] > 0\
                                       else " %sk"%(self.V_latex)
                else:
                    V_name, V_latex, V_sym , V_f\
                            = convert_to_function(V, scale_by_k=True)
                    tmp = np.complex((V_f(2.)))
                    self.V = V_f
                    self.V_x = scale(V_f(self.x), 15)
                    self.V_name = V_name
                    self.V_latex = V_latex
                    self.U_t = Unitary_Operator_1D(V_f)
            except (TypeError, AttributeError,
                    SyntaxError, ValueError, NameError) as E:
                print(E)
        elif isinstance(V, np.ndarray):
            self.V = None
            self.V_x = scale(V, 15)
            self.V_name = "V(x)"
            self.V_latex = "$V(x)$"
            self.U_t = Unitary_Operator_1D(V)
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
            self.lines[8].set_text("E = %.0f"%(V_max))
            self.lines[9].set_text("E = %.0f"%(-V_max))
        elif np.amax(self.V_x < 0):
            V_max = np.abs(np.amin(self.V_x[1:-2]))
            self.lines[4].set_ydata(self.V_x/
                                    V_max*self.bounds[-1]*0.95)
            V_max *= self._scale
            self.lines[8].set_text("E = %.0f"%(V_max))
            self.lines[9].set_text("E = %.0f"%(-V_max))
        else:
            V_max = self.bounds[-1]*0.95*self._scale
            self.lines[4].set_ydata(self.x*0.0)
            self.lines[8].set_text("E = %.0f"%(V_max))
            self.lines[9].set_text("E = %.0f"%(-V_max))

        #Update the text display
        if (self.V_latex.replace(".","").isnumeric() and
            (float(self.V_latex) == 0.)):
            self.lines[5].set_text(
            "$H = %s$, \n%s"%(self._KE_ltx, self._lmts_str))
            self._main_msg = self.lines[5].get_text()
        elif(self.V_latex[1] == "-"):
            self.lines[5].set_text(
            "$H = %s $%s, \n%s"%(self._KE_ltx,
            self.V_latex, self._lmts_str))
            self._main_msg = self.lines[5].get_text()
        else:
            self.lines[5].set_text(
            "$H = %s + $%s, \n%s"%(self._KE_ltx,
            self.V_latex, self._lmts_str))
            self._main_msg = self.lines[5].get_text()

    def display_probability(self, *args):
        """
        Show only the probability density |\psi(x)|^2.
        """
        self._display_probs = True
        self.lines[1].set_linewidth(1.25)
        self.lines[2].set_alpha(0.)
        self.lines[3].set_alpha(0.)
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
        self.lines[0].set_text("—— $|\psi(x)|$")
        #self.lines[0].set_text("—— |Ψ(x)|")
        self.lines[6].set_alpha(1.)
        self.lines[7].set_alpha(1.)

    def measure_energy(self, *args):
        """
        Measure the energy. This collapses the wavefunction
        to the most probable energy eigenstate.
        """
        if not hasattr(self.U_t, "energy_eigenvalues"):
            self.U_t.Set_Energy_Eigenstates()
        EE = np.sort(np.real(self.U_t.energy_eigenvalues))
        EEd = {E:(i + 1) for i, E in enumerate(EE)}
        E = self.psi.set_to_eigenstate(
            self.U_t.energy_eigenvalues,
            self.U_t.energy_eigenstates)
        n = ordinate(str(EEd[np.real(E)]))
        self._msg = "Energy E = %s\n(%s energy level)"%(
        str(np.round(np.real(E), 1)), n)
        self._msg_i = 50

    def lower_energy_eigenstate(self, *args):
        """
        Go to a lower energy eigenstate
        """
        if not hasattr(self.U_t, "energy_eigenvalues"):
            self.U_t.Set_Energy_Eigenstates()
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
        self.U_t._nE -= 1 if self.U_t._nE > 0 else 0
        n = self.U_t._nE
        E = np.real(self.U_t.energy_eigenvalues[n])
        self.psi.x = self.U_t.energy_eigenstates.T[n]
        self.psi.normalize()
        n = ordinate(str(n + 1))
        self._msg = "Energy E = %s\n(%s energy level)"%(
        str(np.round(np.real(E), 1)), n)
        self._msg_i = 50

    def  higher_energy_eigenstate(self, *args):
        """
        Go to a higher energy eigenstate
        """
        #Copy and paste from the measure_energy method
        if not hasattr(self.U_t, "energy_eigenvalues"):
            self.U_t.Set_Energy_Eigenstates()
        if not hasattr(self.U_t, "_nE"):
            self.U_t._nE = 0
            self._nE = -1
            ind = np.argsort(
                np.real(self.U_t.energy_eigenvalues))
            eigvects = np.copy(self.U_t.energy_eigenstates).T
            eigvals = np.copy(self.U_t.energy_eigenvalues)
            for i, j in enumerate(ind):
                eigvals[i] = self.U_t.energy_eigenvalues[j]
                eigvects[i] = self.U_t.energy_eigenstates.T[j]
            self.U_t.energy_eigenvalues = eigvals
            self.U_t.energy_eigenstates = eigvects.T
        n_eigvals = len(self.U_t.energy_eigenvalues)
        self.U_t._nE += 1 if self.U_t._nE < n_eigvals - 1 else 0
        n = self.U_t._nE
        E = np.real(self.U_t.energy_eigenvalues[n])
        self.psi.x = self.U_t.energy_eigenstates.T[n]
        self.psi.normalize()
        n = ordinate(str(n + 1))
        self._msg = "Energy E = %s\n(%s energy level)"%(
        str(np.round(np.real(E), 1)), n)
        self._msg_i = 50

    def measure_position(self, *args):
        """
        Measure the position. This collapses the wavefunction
        to the most probable position eigenstate.
        """
        x = self.psi.set_to_eigenstate(self.x, self.U_t.I)
        self._msg = "Position x = %s"%(str(np.round(x, 3)))
        self._msg_i = 50

    def measure_momentum(self, *args):
        """
        Measure the momentum. This collapses the wavefunction
        to the most probable momentum eigenstate.
        """
        p = self.psi.set_to_momentum_eigenstate()
        freq = str(int(p*(1/(2*np.pi*self.hbar/self.L))))
        self._msg = "Momentum p = %s\n(k = %s)"%(
        str(np.round(p, 3)), freq)
        self._msg_i = 50

    def set_m(m, *args):
        """
        Change the mass of the particle
        """
        self.m = m
        self.psi.m = m
        self.U_t.m = m
        self.U_t = self.set_unitary(self.V_x)

    def _change_constants(self, hbar, *args):
        """
        Change constants
        """
        self.hbar = hbar
        self.psi.hbar = hbar
        self.U_t.hbar = hbar
        self.U_t = self.set_unitary(self.V_x)

    def toggle_expectation_values(self):
        """
        toggle whether to show expectation values
        or not
        """
        if self._show_exp_val == True:
            self.lines[5].set_text(self._main_msg)
            self.lines[5].set_position(self._msg_pos)
        else:
            self._msg_pos = self.lines[5].get_position()
            x, y = self._msg_pos
            #If not showing any fps stats or stdevs
            #change y to y*0.8
            #if showing stdevs and avgs change to  
            #If also showing fps stats, change to 0.5
            #If showing both fps stats and stdevs change to 0.1
            self.lines[5].set_position((x, y*0.4))

        self._show_exp_val = not self._show_exp_val


    def _init_plots(self):
        """
        Initialize the animation.
        In this method we:
        - initialize the required matplotlib objects
        - determine the boundaries of our plot
        """

        # Please note, if you change attributes L and x0 in the
        # base Constants class, you may also need to change:
        # - The location of text labels

        # Make matplotlib figure object
        self.figure = plt.figure(dpi=self._dpi)

        # Make a subplot object
        self.ax = self.figure.add_subplot(1,1,1)

        # Add a grid
        #self.ax.grid(linestyle="--")

        # Set the x limits of the plot
        xmin = self.x[0]
        xmax = self.x[-1]
        xrange = xmax - xmin
        self.ax.set_xlim(self.x[0] - 0.02*xrange,
                         self.x[-1] + 0.02*xrange)
        self.ax.set_xlabel("x")

        # Set the y limits of the plot
        ymax = np.amax(np.abs(self.psi.x))
        #ymin=np.amin(np.abs(psi.x))
        ymin = -ymax
        yrange = ymax - ymin

        self.ax.get_yaxis().set_visible(False)

        self.ax.set_ylim(ymin-0.1*yrange, ymax+0.1*yrange)

        # Set inital plots with ax.plot.
        # They return the line object which controls the appearance
        # of their plots.
        # Note that the number naming of the variables is not in any
        # logical order.
        # This is due to oversight.
        # TODO: Fix the ordering

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
                              #color="blue",
                              animated=True,
                              #label=r"$Re(\psi(x))$",
                              linewidth=0.5)
        line3, = self.ax.plot(self.x, np.imag(self.psi.x),
                              "-",
                              #color="orange",
                              animated=True,
                              #label=r"$Im(\psi(x))$",
                              linewidth=0.5)
        line1, = self.ax.plot(self.x, np.abs(self.psi.x),
                              animated=True,
                              #label=r"$|\psi(x)|$",
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
                             #animated=True
                             )


        line0 = self.ax.text((xmax-xmin)*0.01 + xmin,
                             ymin + (ymax-ymin)*0.05,
                             #"—— |Ψ(x)|",
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

        #bottom = np.linspace(ymin, ymin, self.N)
        #self.fill = self.ax.fill_between(self.x, bottom,
        #                                 self.V_x/np.amax(self.V_x[1:-2]),
        #                                 color="gray", alpha=0.05)

        # Store each line in a list.
        self.lines = [line0, line1, line2, line3,
                      line4, line5,
                      line6, line7
                      #line8
                      ]

        #Another round of setting up and scaling the line plots ...
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

        #Manually plot gird lines
        maxp = self.bounds[-1]*0.95
        self.ax.plot([self.x0, self.x0+self.L], [0., 0.],
                     color="gray", linewidth=0.5, linestyle = "--")
        self.ax.plot([self.x0, self.x0+self.L], [maxp, maxp],
                     color="gray", linewidth=0.5, linestyle = "--")
        self.ax.plot([self.x0, self.x0+self.L], [-maxp, -maxp],
                     color="gray", linewidth=0.5, linestyle = "--")
        #self.ax.plot([0, 0], [-self.bounds[-1], self.bounds[-1]],
        #             color="gray", linewidth=0.5, linestyle = "--")

        #Show where the potential is zero
        self.ax.text(xmax*0.8, 0.03, "E = 0", color="gray", fontdict={'size':8})
        self.lines.append(self.ax.text
                          (xmax*0.7, maxp*0.92,
                           "E = %.0f"%(V_max),
                           color="gray", fontdict={'size':8}))
        self.lines.append(self.ax.text
                          (xmax*0.68, -maxp*0.96,
                           "E = %.0f"%(-V_max),
                           color="gray", fontdict={'size':8}))

        self._main_msg = self.lines[5].get_text()


    def _animate(self,i):
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

        # Set probability density or absolute value of wavefunction
        if self._display_probs:
            # An underflow error occurs here after
            # measuring the position.
            # Just ignore this for now.
            try:
                self.lines[1].set_ydata(
                    np.real(np.conj(self.psi.x)*self.psi.x)/3)
            except FloatingPointError as E:
                print(E)
        else:
            self.lines[1].set_ydata(np.abs(self.psi.x))

        # Set real and imaginary values
        self.lines[2].set_ydata(np.real(self.psi.x))
        self.lines[3].set_ydata(np.imag(self.psi.x))

        # Find fps stats
        t0, tf = self.t_perf
        self.ticks += 1
        self.fps = int(1/(tf - t0 + 1e-30))
        if self.ticks > 1:
            self.fps_total += self.fps
        self.avg_fps = int(self.fps_total/(self.ticks))

        # Output temporary text messages
        if (self._msg_i > 0):
            self.lines[5].set_text(self._msg)
            self._msg_i += -1
        elif (self._msg_i == 0):
            t0, tf = self.t_perf
            self._msg_i += -1
            self.lines[5].set_text(self._main_msg)

        elif (self._show_exp_val and self._msg_i < 0):
            if not hasattr(self.U_t, "energy_eigenvalues"):
                self.U_t.Set_Energy_Eigenstates()
            x_mean, x_sigma = self.psi.avg_and_std(self.x, self.U_t.I)
            p_mean, p_sigma = self.psi.p_avg_and_std()
            E_mean, E_sigma = self.psi.avg_and_std(
                self.U_t.energy_eigenvalues,
                self.U_t.energy_eigenstates
                )
            self.lines[5].set_text(
                "t = %f\n"
                #"fps = %i\n"
                #"avg_fps = %i\n"
                "<x> = %.2f\n"
                "<p> = %.2f\n"
                "<E> = %.0f\n"
                "σₓ = %.2f\n"
                "σₚ = %.2f\n"
                "σᴇ = %.0f"%(
                    self._t,
                    #self.fps,
                    #self.avg_fps,
                    x_mean,
                    p_mean,
                    E_mean,
                    x_sigma,
                    p_sigma,
                    E_sigma
                    )
                )

        return (self.lines)

    def animation_loop(self):
        """Produce all frames of animation.
        """
        self.main_animation=animation.FuncAnimation\
                (self.figure, self._animate, blit=True,
                 interval=1
                 )

    def get_fpi(self):
        """Get the number of time evolutions per animation
        frame"""
        return self.fpi

    def set_fpi(self, new_fpi):
        """Set the number of time evolutions per animation frame
        """
        self.fpi = new_fpi

if __name__ == "__main__":

    from matplotlib import interactive
    interactive(True)

    C = Constants()
    x = np.linspace(C.x0, (C.L + C.x0), C.N)
    V = (x)**2/2
    #psi = np.exp(-0.5*((x-0.25)/0.05)**2)
    psi = np.cos(3*np.pi*x/C.L)
    Ani = QM_1D_Animation(function=psi, potential=V)
    Ani.animation_loop()
    #plt.legend()
    plt.show()
