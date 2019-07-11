
# 1D Time-Independent Quantum Mechanics Applet Using Python and Tkinter

This repository contains an applet that simulates the dynamics of a quantum particle inside a reshapable 1D time-independent potential of finite extent. The time evolution of this quantum mechanical particle is given by
|&psi;(t)> = U(t)|&psi;(0)>,
where |&psi;> is the wavefunction vector ket of the particle and U(t) is the unitary time evolution matrix. The matrix U(t) can be found on a computer by using the Crank-Nicolson method. A derivation of this method is contained in [Exercise 9.8 of Mark Newmann's Computational Physics](http://www-personal.umich.edu/~mejn/cp/exercises.html) for the free particle case. It is straightfoward to generalize this to a particle bounded in any energy-preserving potential.

The following modules are used: Numpy for numerical computation, Sympy for symbolic computation, Matplotlib for plotting, and Tkinter for implementing the GUI.

## Instructions:

Make sure you have the latest version of Python with Numpy, Matplotlib, Sympy, and Tkinter installed. The code for this applet is obtained by cloning this repository or downloading it. Ensure that all source code are in the same directory. Now run `Applet.py` - you should get something like:

<img src="https://raw.githubusercontent.com/marl0ny/1D-Time-Independent-Quantum-Mechanics-Applet-Using-Python-and-Tkinter/master/Coherent%20State%20SHO.gif" />


The main window displays our wavefunction &psi;(x), where we show its real part, imaginary part, and absolute value. At the top left corner of the window is an info box, which displays the Hamiltonian H and the boundaries of our quantum system. At the bottom of the window is the x-scale, which extends from -0.5 to 0.5. Note that this scale and all other units of measurement are not in metric units, but in [natural units](https://en.wikipedia.org/wiki/Natural_units), where all fundamental constants are set to one.

You can disturb and reshape the wavefunction by clicking on it:

<img src="https://raw.githubusercontent.com/marl0ny/1D-Time-Independent-Quantum-Mechanics-Applet-Using-Python-and-Tkinter/master/making%20waves.gif" />

The wavefunction &psi;(x) is not physically observable, but its probability density \*&psi;(x)&psi;(x) is, which you can switch to by clicking `View Probability Distribution`. To measure observables, click on the `Measure Position`, `Measure Momentum`, and `Measure Energy` buttons. You can also enter a new wavefunction &psi;(x) or potential V(x) in the `Enter Wavefunction` or `Enter Potential V(x)` text boxes. Please ensure that your entry is a function of x and that you hit the return key after. To alter the speed of the simulation, move the `Animation Speed` slider. To close the application, press `QUIT`.

If you just want a plain Matplotlib animation, you can either run `Animate.py` or import it separately, either through the command line or in a new file. If you import it, make sure that you are running Matplotlib in interactive mode, and that you create an instance of the animation object `QM_1D_Animation`. You can then measure observables by using its methods `measure_energy()`, `measure_position()`, and `measure_momentum()`. To switch between a view of the wavefunction itself or its probability density, use the methods `display_probability()` or `display_wavefunction()`. To change the wavefunction or potential, use `set_wavefunction([new wavefunction])` and `set_unitary([new_potential])`, where `new_wavefunction` and `new_potential` can either be a string or a Numpy array.

The actual physics are contained in the classes `Wavefunction_1D` and `Unitary_Operator_1D`, which represent the wavefunction and unitary operator. These are in the file `QM_1D_TI.py`.

## References

Newman, M. (2013). Partial differential equations. In <em>[Computational Physics](http://www-personal.umich.edu/~mejn/cp/)</em>, chapter 9. CreateSpace Independent Publishing Platform.
