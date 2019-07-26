

# 1D Time-Independent Quantum Mechanics Applet Using Python and Tkinter

This repository contains an applet that simulates the dynamics of a quantum particle inside a reshapable 1D time-independent potential of finite extent. The time evolution of this quantum mechanical particle is given by
|&psi;(t)> = U(t)|&psi;(0)>,
where |&psi;> is the wavefunction vector ket of the particle and U(t) is the unitary time evolution matrix. The matrix U(t) can be found on a computer by using the Crank-Nicolson method. A derivation of this method is contained in [Exercise 9.8 of Mark Newmann's Computational Physics](http://www-personal.umich.edu/~mejn/cp/exercises.html) for the free particle case. It is straightfoward to generalize this to a particle bounded in any energy-preserving potential.

The following modules are used: Numpy for numerical computation, Sympy for symbolic computation, Matplotlib for plotting, and Tkinter for implementing the GUI.

## Instructions:

Make sure that you have at least Python 3.6.5 with Numpy, Matplotlib and Sympy installed.  You can install these by running  `pip3 install numpy matplotlib sympy`.  You may also optionally install Numba, which speeds up the performance of this applet.  To install Numba, run `pip3 install numba`.

The code for this applet is obtained by cloning this repository or downloading it. Ensure that all source code are in the same directory. Now run `Applet.py` - you should get something like:

<img src="https://raw.githubusercontent.com/marl0ny/1D-Time-Independent-Quantum-Mechanics-Applet-Using-Python-and-Tkinter/master/Images/Coherent State SHO 2.gif" />


The main window displays the wavefunction &psi;(x) with the outline of the potential V(x) in the background. At the top left corner of the window is an info box, which displays the Hamiltonian H and the boundaries of our quantum system. At the bottom of the window is the x-axis, which extends from -0.5 to 0.5. Note that this scale and all other units of measurement are not in metric units, but in [natural units](https://en.wikipedia.org/wiki/Natural_units), where all fundamental constants are set to one.

You can disturb and reshape the wavefunction by clicking on it. You can also use the mouse to reshape the potential V(x). To do this click on the `Mouse` drop-down menu and select `Reshape Potential V(x)`, then draw a new shape for the potential:

<img src="https://raw.githubusercontent.com/marl0ny/1D-Time-Independent-Quantum-Mechanics-Applet-Using-Python-and-Tkinter/master/Images/demo.gif" />

Now the wavefunction &psi;(x) is not physically observable, but its probability density \*&psi;(x)&psi;(x) is, which you can switch to by clicking `View Probability Distribution`. To measure observables, click on the `Position`, `Momentum`, and `Energy` buttons. To view the expectation values and standard deviation of observables, right click and select `Toggle Expectation Values`. To cycle between different stationary states, press the up and down keys, or right click and select either `Higher Stationary State` or `Lower Stationary State`. You can type in a new wavefunction &psi;(x) and potential V(x) in `Enter Wavefunction ψ(x)` and `Enter Potential V(x)` text boxes. You can also select a preset potential in the `Choose Preset Potential V(x)` drop-down menu. To clear the wavefunction, press the `Clear Wavefunction` button. To alter the speed of the simulation, move the `Animation Speed` slider. To close the application, press `QUIT`.

If you just want a plain Matplotlib animation, you can either run `Animate.py` or import it separately, either through the command line or in a new file. If you import it, make sure that you are running Matplotlib in interactive mode, and that you create an instance of the animation object `QM_1D_Animation`. You can then measure observables by using its methods `measure_energy()`, `measure_position()`, and `measure_momentum()`. To switch between a view of the wavefunction itself or its probability density, use the methods `display_probability()` or `display_wavefunction()`. To change the wavefunction or potential, use `set_wavefunction([new wavefunction])` and `set_unitary([new_potential])`, where `new_wavefunction` and `new_potential` can either be a string or a Numpy array.

The actual physics are contained in the classes `Wavefunction_1D` and `Unitary_Operator_1D`, which represent the wavefunction and unitary operator. These are in the module `QM_1D_TI`.

## References

Newman, M. (2013). Partial differential equations. In <em>[Computational Physics](http://www-personal.umich.edu/~mejn/cp/)</em>, chapter 9. CreateSpace Independent Publishing Platform.
