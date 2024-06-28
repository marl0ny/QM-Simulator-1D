"""
Single-Particle 1D Quantum Mechanics module.
"""
import numpy as np
from .constants import Constants


class Wavefunction1D(Constants):
    """
    Wavefunction class in 1D.

    Attributes:
    x [np.ndarray]: wavefunction in the position basis
    """

    def __init__(self, waveform):

        super().__init__()

        if callable(waveform):

            # Depending on the version of sympy,
            # passing arrays to lambdified functions
            # produces an error
            try:
                self.x = waveform(np.linspace(self.x0,
                                              (self.L + self.x0),
                                              self.N))
            except:
                tmpx = np.linspace(self.x0, (self.L + self.x0), self.N)
                self.x = np.array([waveform(x) for x in tmpx])

            # This is a quick fix to the issue where the
            # lambdify function returns a single 0 for all
            # input, which occurs when strings of the
            # form "0*x" are inputed.
            try:
                len(self.x)
            except TypeError as E:
                print(E)

        elif isinstance(waveform, np.ndarray):
            self.x = waveform

    def normalize(self):
        """
        Normalize the wavefunction
        """

        try:
            self.x = self.x/np.sqrt(
                np.trapz(np.conj(self.x)*self.x, dx=self.dx))
        except FloatingPointError as E:
            print(E)

    @property
    def p(self):
        """
        The wavefunction in the momentum basis.
        """
        return np.fft.fftshift(np.fft.fft(self.x)/(self.N/10))

    def expectation_value(self, eigenvalues, eigenstates):
        """
        Find the expectation value of the wavefunction
        with respect to the eigenvalues and eigenstates
        of a Hermitian Operator
        """
        try:
            prob = np.abs(np.dot(self.x, eigenstates))**2
            if np.max(prob) != 0.0:
                prob = prob/np.sum(prob)
        except FloatingPointError as E:
            return 0.
            print(E)
        return np.sum(np.dot(np.real(eigenvalues), prob))

    def expected_momentum(self):
        """
        Find the momentum expectation value of the
        wavefunction
        """
        F = np.fft.fft(self.x)
        prob = np.abs(F)**2
        if np.max(prob) != 0.0:
            prob = prob/np.sum(prob)
        freq = np.fft.fftfreq(self.N, d=self.dx)
        p = 2*np.pi*freq*self.hbar/self.L
        return np.dot(p, prob)

    def average_and_standard_deviation(self, eigenvalues, eigenstates):
        """
        Find the expectation value and the standard deviation
        of the wavefunction with respect to the eigenvalues and
        eigenstates of a Hermitian Operator
        """
        try:
            prob = np.abs(np.dot(self.x, eigenstates))**2
            if (np.max(prob) != 0.):
                prob = prob/np.sum(prob)

            expval = np.sum(np.dot(np.real(eigenvalues), prob))
            expval2 = np.sum(np.dot(np.real(eigenvalues)**2, prob))
            sigma = np.sqrt(expval2 - expval**2)
            return (expval, sigma)

        except FloatingPointError as E:
            print(E)
            return (0., 0.)

    def momentum_average_and_standard_deviation(self):
        """
        Find the expectation value and the standard deviation
        of the wavefunction with respect to the eigenvalues and
        eigenstates of the momentum operator
        """
        F = np.fft.fft(self.x)
        prob = np.abs(F)**2
        if np.max(prob) != 0.0:
            prob = prob/np.sum(prob)
        freq = np.fft.fftfreq(self.N, d=self.dx)
        p = 2*np.pi*freq*self.hbar/self.L

        expval = np.sum(np.dot(p, prob))
        expval2 = np.sum(np.dot(p**2, prob))
        sigma = np.sqrt(expval2 - expval**2)
        return (expval, sigma)

    def set_to_momentum_eigenstate(self):
        """
        Set the wavefunction to an allowable
        momentum eigenstate.
        Return the momentum eigenvalue.

        The momentum eigenstates may not
        actually be defined for a particle in
        a box. See, for example, these questions:

        https://physics.stackexchange.com/q/66429
        [Question by JLA:
         physics.stackexchange.com/
         users/10964/jla]

        https://physics.stackexchange.com/q/45498
        [Question by Benji Remez:
         physics.stackexchange.com/
         users/12533/benji-remez]

         In any case, we Fourier transform the wavefunction, square
         it and find the most likely amplitude, and then inverse Fourier
         transform only this amplitude to obtain the momentum eigenstate.
        """
        F = np.fft.fft(self.x)
        prob = np.abs(F)**2
        if np.max(prob) != 0.0:
            prob = prob/np.sum(prob)
        freq = np.fft.fftfreq(self.N, d=self.dx)
        choice = np.random.choice(a=freq, size=1,
                                  p=prob, replace=False)
        k = choice[0]
        if k == 0.0:
            self.x = np.ones([self.N])
            self.normalize()
            p = 2*np.pi*k*self.hbar/self.L
            return p
        freq = np.array(
            [(0. if f != k else f) for f in freq])
        F = freq*F
        self.x = np.fft.ifft(F)
        self.normalize()
        p = 2*np.pi*k*self.hbar/self.L
        return p

    def set_to_eigenstate(self, eigenvalues, eigenstates, smear=False):
        """
        Set the wavefunction to an eigenstate of
        any operator. Given the eigenvalues and eigenvectors
        of the operator, reset the wavefunction to the
        most probable eigenstate, then return the
        eigenvalue that corresponds to this eigenstate.
        """

        prob = np.abs(np.dot(self.x, eigenstates))**2
        if np.max(prob) != 0.0:
            prob = prob/np.sum(prob)

        a = [i for i in range(len(prob))]
        choice = np.random.choice(a=a, size=1, p=prob, replace=False)
        self.x = (eigenstates.T[[choice[0]]][0])
        if smear == True:
            for i in range(1, 3):
                if choice[0] - i >= 0 and choice[0] + i < self.N: 
                    self.x += ((eigenstates.T[[choice[0] + i]][0])*
                               np.exp(-(i/(1))**2.0/2.0))
                    self.x += ((eigenstates.T[[choice[0] - i]][0])*
                               np.exp(-(i/(1))**2.0/2.0))

        self.normalize()

        return eigenvalues[choice[0]]


class UnitaryOperator1D(Constants):
    """A unitary operator that dictates time evolution
    of the wavefunction.

    Attributes:
    U [np.ndarray]: Unitary time evolution matrix
    id [np.ndarray]: Identity matrix
    energy_eigenstates: Energy eigenstates of the
    [np.ndarray]        Hamiltonian
    energy_eigenvalues: The corresponding energy
    [np.ndarray]        to each eigenstate

    """

    def __init__(self, Potential):
        """Initialize the unitary operator.
        """

        # The unitary operator can be found by applying the Cranck-Nicholson
        # method. This method is outlined in great detail in exercise 9.8 of
        # Mark Newman's Computational Physics. Here is a link to a
        # page containing all problems in his book:
        # http://www-personal.umich.edu/~mejn/cp/exercises.html

        # Newman, M. (2013). Partial differential equations.
        # In Computational Physics, chapter 9.
        # CreateSpace Independent Publishing Platform.

        # TODO: This can be optimized.

        super().__init__()

        if isinstance(Potential, np.ndarray):
            V = Potential

        elif callable(Potential):
            x = np.linspace(self.x0, (self.L + self.x0), self.N)
            V = np.array([Potential(xi) for xi in x])

        V *= self._scale

        # Get constants
        m, hbar, e, L, N, dx, dt = self._get_constants()

        # Initialize A and B matrices
        A = np.zeros([N, N], np.complex128)
        B = np.zeros([N, N], np.complex128)

        # \Delta t \frac{i \hbar}{2m \Delta x^2}
        K = (dt*1.0j*hbar)/(4*m*dx**2)

        # \frac{\Delta t i \hbar}{2}
        J = (dt*1.0j)/(2*hbar)

        # Initialize the constant,
        # nonzero elements of the A and B matrices
        a1 = 1 + 2*K
        a2 = -K
        b1 = 1 - 2*K
        b2 = K

        # Construct the A and B matrices
        for i in range(N-1):
            A[i][i] = a1 + J*V[i]
            B[i][i] = b1 - J*V[i]
            A[i][i+1] = a2
            A[i+1][i] = a2
            B[i][i+1] = b2
            B[i+1][i] = b2
        A[N-1][N-1] = a1 + J*V[N-1]
        B[N-1][N-1] = b1 - J*V[N-1]

        # Obtain U
        self.U = np.dot(np.linalg.inv(A), B)
        # self.U = np.matmul(np.linalg.inv(A), B)

        # The identity operator is what the unitary matrix
        # reduces to at time zero. Also,
        # since the wavefunction and all operators are
        # in the position basis, the identity matrix
        # is the position operator.
        self.id = np.identity(len(self.U[0]), np.complex128)

        # Get the Hamiltonian from the unitary operator
        # and aquire the energy eigenstates.
        # self.set_energy_eigenstates()

    def __call__(self, wavefunction):
        """Call this class
        on a wavefunction to time evolve it.
        """
        try:
            wavefunction.x = np.matmul(self.U, wavefunction.x)
        except FloatingPointError:
            pass
        # wavefunction.x = np.dot(self.U, wavefunction.x)

    def _set_HU(self):
        """Set HU (the Hamiltonian times the unitary operator).
        Note that HU is not Hermitian.
        """
        # The Hamiltonian H is proportional to the
        # time derivative of U times its inverse
        self._HU = (0.5*1.0j*self.hbar/self.dt)*(self.U - np.conj(self.U.T))
        # self._HU =(1.0j*self.hbar/self.dt)*(self.U - self.id)

    def set_energy_eigenstates(self):
        """Set the eigenstates and energy eigenvalues.
        """
        self._set_HU()

        eigvals, eigvects = np.linalg.eigh(self._HU)
        eigvects = eigvects.T
        eigvals = np.sign(np.real(eigvals))*np.abs(eigvals)

        tmp_dict = {}
        for i in range(len(eigvals)):
            E = np.round(eigvals[i], 6)
            if E in tmp_dict:
                tmp_dict[E] = np.add(eigvects[i], tmp_dict[E])
            else:
                tmp_dict[E] = eigvects[i]

        eigvals, eigvects = tmp_dict.keys(), tmp_dict.values()
        self.energy_eigenvalues = np.array(list(eigvals))
        self.energy_eigenstates = np.array(list(eigvects), np.complex128).T
