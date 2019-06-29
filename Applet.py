from Animate import *
from QM_1D_TI import _constants
import tkinter as tk

np.seterr(all='raise')
# Make numpy raise errors instead of warnings.

# TODO:
# -Add citations: Newman's book.
# -Make some optimizations to the Unitary_Operator_1D class
# -Clean up the QM_1D_Animation class.
# FUTURE FEATURES:
# -Make it possible to change the mass of the particle and other constants
# -Allow reshaping of the potential using mouse input
# -Make a visualization of the energy levels of the potential
# -Make the wavefunction viewable in momentum space
# -Add time dependance

class Applet(QM_1D_Animation):
    """
    QM Applet using Tkinter.

    Attributes:
    window [tkinter.Tk]
    canvas [matplotlib.backends.backend_tkagg.FigureCanvasTkAgg]
    menu [tk.Menu]
    enter_function_label [tk.Label]
    enter_function [tk.Entry]
    enter_potential_label [tk.Label]
    enter_potential [tk.Entry]
    measure_position_button [tk.Button]
    measure_momentum_button [tk.Button]
    measure_energy_button [tk.Buton]
    slider_speed_label [tk.Label]
    slider_speed [tk.Scale]
    quit_button [tk.Button]
    """

    def __init__(self):
        """
        Initializer.
        """

        self.window = tk.Tk()
        self.window.title("Bounded Wavefunction in 1D")
        self.window.configure(background="white")

        #Set plot resolution
        #width = self.window.winfo_screenwidth()
        #print(width)
        #dpi = 250

        C = _constants()
        x = np.linspace(C.x0, C.L + C.x0, C.N)

        #Defaults
        V = "(x)**2/2"
        psi = np.exp(-0.5*((x-0.25)/0.05)**2)

        #Other
        #V = rect(8*x)
        #psi = np.exp(-0.5*((x-0.25)/0.05)**2 - 160j*x*np.pi)

        #Initialize the inherited animation object
        QM_1D_Animation.__init__(self, function=psi, potential=V)

        #Canvas

        # A quick example of how to integrate a matplotlib animation into a
        # tkinter gui is given by this Stack Overflow question and answer:

        # https://stackoverflow.com/q/21197728
        # [Question by user3208454:
        # https://stackoverflow.com/users/3208454/user3208454]

        # https://stackoverflow.com/a/21198403
        # [Answer by HYRY:
        # https://stackoverflow.com/users/772649/hyry]

        self.canvas = \
        backend_tkagg.FigureCanvasTkAgg(self.figure, master=self.window)
        self.canvas.get_tk_widget().grid(row=0,
                                         column=0,
                                         rowspan=12,
                                         columnspan=3)
        self.canvas.get_tk_widget().bind("<B1-Motion>",
                         self.update_wavefunction_by_sketch)
        self.canvas.get_tk_widget().bind("<ButtonRelease-1>",
                         self.update_wavefunction_by_sketch)

        #Right click Menu
        self.menu = tk.Menu(self.window, tearoff=0)
        self.menu.add_command(label="quit", command=self.quit)
        self.window.bind("<ButtonRelease-3>", self.popup_menu)

        #Show Probability Distribution/Wavefunction
        self.change_view = tk.Button(self.window,
                                            text
                                            ='View Probability Distribution',
                                            command=lambda:
                                     [self.display_probability(),
                                      self.change_view.config(
                                          text='View Wavefunction')] if
                                     (self._display_probs == False) else
                                        [self.display_wavefunction(),
                                         self.change_view.config(
                                             text=
                                             'View Probability Distribution')]
                                     )
        self.change_view.grid(row=1, column=3, padx=(10,10))

        #Measure position button
        self.measure_position_button = tk.Button(self.window,
                                               text='Measure Position',
                                               command=self.measure_position)
        self.measure_position_button.grid(row=2, column=3, padx=(10,10))

        #Measure momentum button
        self.measure_momentum_button = tk.Button(self.window,
                                               text='Measure Momentum',
                                               command=self.measure_momentum)
        self.measure_momentum_button.grid(row=3, column=3, padx=(10,10))

        #Measure energy button
        self.measure_energy_button = tk.Button(self.window,
                                               text='Measure Energy',
                                               command=self.measure_energy)
        self.measure_energy_button.grid(row=4, column=3, padx=(10,10))

        #Wavefunction entry field
        self.enter_function_label=\
                        tk.Label(self.window, text="Enter Wavefunction")
        self.enter_function_label.grid(row=5, column=3, padx=(10,10))

        self.enter_function=tk.Entry(self.window)
        self.enter_function.bind("<Return>", self.update_wavefunction_by_name)
        self.enter_function.grid(row=6, column=3, padx=(10,10))

        #Potential function entry field
        self.enter_potential_label=\
                        tk.Label(self.window, text="Enter Potential V(x)")
        self.enter_potential_label.grid(row=7, column=3, padx=(10,10))

        self.enter_potential=tk.Entry(self.window)
        self.enter_potential.bind("<Return>", self.update_potential_by_name)
        self.enter_potential.grid(row=8, column=3, padx=(10,10))

        #Animation speed slider
        self.slider_speed_label=tk.Label(self.window, text="Animation Speed")
        self.slider_speed_label.grid(row=9, column=3,padx=(10,10))

        self.slider_speed=tk.Scale(self.window, from_=0, to=8,
                                   orient=tk.HORIZONTAL,
                                   length=200,
                                   command=self.change_animation_speed)
        self.slider_speed.grid(row=10, column=3, padx=(10,10))
        self.slider_speed.set(1)

        #Quit button
        self.quit_button=tk.Button\
                          (self.window, text='QUIT',command=self.quit)
        self.quit_button.grid(row=11, column=3)

        self.animation_loop()

    def popup_menu(self, event):
        """
        popup menu upon right click.
        """
        self.menu.tk_popup(event.x_root, event.y_root, 0)

    def update_wavefunction_by_name(self, event):
        """
        Update the wavefunction given entry input.
        """
        self.set_wavefunction(self.enter_function.get())

    def update_wavefunction_by_sketch(self, event):
        """
        Update the wavefunction given mouse clicks
        on to the canvas. Note that the animation
        can still run if the wavefunction is changed.
        """
        x, y = self.locate_mouse(event)
        if (self._display_probs):
            psi2_new = change_array(
                self.x, self.psi.x*np.conj(self.psi.x)/3, x, y)
            self.set_wavefunction(np.sqrt(3*psi2_new),
                                  normalize=False)
        else:
            self.set_wavefunction(change_array(
                self.x, self.psi.x, x, y), normalize=False)

    def update_potential_by_name(self, event):
        """
        Update the potential given entry input.
        """
        self.set_unitary(self.enter_potential.get())

    def change_animation_speed(self, event):
        """
        Change the animation speed.
        """
        self.fpi = self.slider_speed.get()

    def locate_mouse(self, event):
        """
        Locate the position of the mouse with respect to the
        coordinates displayed on the plot axes.
        """

        # Get the dimensions of the plot
        # in terms of x and y
        xmin, xmax, ymin, ymax = self.bounds

        # Get the x and y ranges.
        xrange = xmax - xmin
        yrange = ymax - ymin

        # Switch to "canvas coordinates". These are the locations of
        # event.x and event.y in terms of pixels with respect to the origin at
        # the bottom left corner of the canvas.
        x_canvas = event.x
        y_canvas = self.canvas.get_tk_widget().winfo_height() - event.y

        # These are the canvas coordinate locations of
        # where the axes begin (pxi, pyi), their intersection (px0, py0),
        # and where they end (pxf, pyf). In order to find these, uncomment
        # the print statements right before this function's return statement
        # and see what they output when clicking on the canvas.
        # Please note, these values are dependant on the values of L and x0
        # in the inherited _constants class.
        if (self._dpi == 100):
            pxi, pxf = 78, 576
            pyi, pyf = 53, 422
            px0, py0 = 328, 238
        elif (self._dpi == 120):
            pxi, pxf = 98, 692
            pyi, pyf = 62, 506
            px0, py0 = 395, 283
        elif (self._dpi == 150):
            pxi, pxf = 121, 865
            pyi, pyf = 78, 632
            px0, py0 = 493, 355
        else:
            # Assume they scale linearly (actually they don't).
            # There are probably much better ways of doing this ...

            dpi = self._dpi
            deltaDPI = 20.

            mpxi = (98 - 78)/deltaDPI
            mpxf = (692 - 576)/deltaDPI
            mpyi = (62 - 53)/deltaDPI
            mpyf = (506 - 422)/deltaDPI
            mpx0 = (395 - 328)/deltaDPI
            mpy0 = (283 - 238)/deltaDPI

            bxi = 98 - mpxi*120
            bxf = 692 - mpxf*120
            byi = 62 - mpyi*120
            byf = 506 - mpyf*120
            bx0 = 395 - mpx0*120
            by0 = 283 - mpy0*120

            pxi, pxf = bxi + mpxi*dpi, bxf + mpxf*dpi
            pyi, pyf = byi + mpyi*dpi, byf + mpyf*dpi
            px0, py0 = bx0 + mpx0*dpi, by0 + mpy0*dpi

        # Range of each axis in terms of pixels.
        pxrange = pxf - pxi
        pyrange = pyf - pyi


        # Transfer to "pixel plot coordinates".
        # In order to go from canvas to pixel plot coordinates,
        # the origin is shifted from the bottom left corner
        # of the canvas to the instersection of the plot axes.
        x_pxl_plot = x_canvas - px0
        y_pxl_plot = y_canvas - py0

        # Transfer from these pixel plot coordinates
        # into the x and y coordinates displayed by the plot axes.
        x = x_pxl_plot * (xrange/pxrange)
        y = y_pxl_plot * (yrange/pyrange)

        # Use the following to find the canvas coordinate
        # locations of where the axes intersect and where they end:
        #self.ax.grid()
        #print ("x: %d, y: %d"%(x_canvas, y_canvas))
        #print (self.bounds)
        #print(x,y)

        return x, y

    def quit(self, *event):
        """
        Quit the application.
        Simply calling self.window.quit() only quits the application
        in the command line, while the GUI itself still runs.
        On the other hand, simply calling self.window.destroy()
        destroys the GUI but doesn't give back control of the command
        line. Therefore both need to be called.
        """
        self.window.quit()
        self.window.destroy()

def change_array(x_arr, y_arr, x, y):
    """Given a location x that maps to a value y,
    and an array x_arr which maps to array y_arr, find the closest
    element in x_arr to x. Then, change its corresponding
    element in y_arr with y.
    """

    if (x < x_arr[0]) or (x > x_arr[-1]):
        return y_arr
    #if (y < y_arr[0]) or (y > y_arr[-1]):
    #    return y_arr

    closest_index = np.argmin(np.abs(x_arr - x))
    y_arr[closest_index] = y

    #If len(x) is large, change nearby values as well.
    if (len(x_arr) > 100):
        try:
            for i in range(3):
                y_arr[closest_index + i] = y
                y_arr[closest_index - i] = y
        except:
            pass

    return y_arr

if __name__ == "__main__":

    App = Applet()
    tk.mainloop()
