"""
Single-Particle 1D Quantum Mechanics module.
"""
import numpy as np
from numba import jit, prange, c16, f8
from numba.types import Tuple
from .constants import Constants
# from time import perf_counter

# Global variables
# TODO: Don't do this.
A = np.zeros((512, 512), np.complex128)
B = np.zeros((512, 512), np.complex128)
# _HU = np.zeros((512, 512), np.complex128)


@jit('c16[:](c16[:,:], c16[:])', parallel=True, nogil=True, nopython=True)
def _time_evolve_wavefunction(U, psix):
    return np.dot(U, psix)


@jit('c16[:,:](c16[:,:], c16[:,:])', nogil=True, nopython=True)
def _mat_mult(M1, M2):
    return np.dot(M1, M2)


@jit('void(i8, c16, c16, c16[:], c16[:,:], c16[:,:])',
     parallel=True, nogil=True, nopython=True)
def _fill_AB(N, K, J, V, A, B):

    # Initialize the constant,
    # nonzero elements of the A and B matrices
    a1 = 1 + 2*K
    a2 = -K
    b1 = 1 - 2*K
    b2 = K

    for i in prange(N-1):
        A[i][i] = a1 + J*V[i]
        B[i][i] = b1 - J*V[i]
        A[i][i+1] = a2
        A[i+1][i] = a2
        B[i][i+1] = b2
        B[i+1][i] = b2
    A[N-1][N-1] = a1 + J*V[N-1]
    B[N-1][N-1] = b1 - J*V[N-1]


# @jit('c16[:,:](c16[:], f8, f8, i8, f8, f8)',
#       parallel=True, nogil=True, nopython=True)
def _construct_U(V, m, hbar, N, dx, dt):

    # Initialize A and B matrices
    # A = np.zeros((N, N), np.complex128)
    # B = np.zeros((N, N), np.complex128)

    # \Delta t \frac{i \hbar}{2m \Delta x^2}
    K = (dt*1.0j*hbar)/(4*m*dx**2)

    # \frac{\Delta t i \hbar}{2}
    J = (dt*1.0j)/(2*hbar)

    _fill_AB(N, K, J, V, A, B)

    invA = np.linalg.inv(A)
    return np.dot(invA, B)
    # return _mat_mult(invA, B)


@jit('Tuple((f8[:], c16[:,:]))(c16[:,:])',
     cache=True, nogil=True, nopython=True)
def _get_eig(H):
    return np.linalg.eigh(H)


@jit('c16[:,:](c16, c16[:,:], c16[:,:])',
     parallel=True, nogil=True, nopython=True)
def _mat_diff(s, A, B):
    return s*(A - B)


@jit('c16[:](c16[:], c16[:])', parallel=True, nogil=True, nopython=True)
def _vect_add(a, b):
    return a + b

class Wavefunction1D(Constants):
    """Wavefunction class in 1D.

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
            except Exception as e:
                if "underflow" in str(e).lower():
                    posarr = np.linspace(self.x0, (self.L + 
                                       self.x0), self.N)
                    self.x = []
                    for i in range(len(posarr)):
                        try:
                            self.x.append(waveform(posarr[i]))
                        except:
                            self.x.append(0.0)
                    self.x = np.array(self.x)
                else:
                    tmpx = np.linspace(self.x0, (self.L + self.x0),
                                    self.N)
                    self.x = np.array([waveform(x) for x in tmpx])

            # This is a quick fix to the issue where the
            # lambdify function returns a single 0 for all
            # input, which occurs when strings of the
            # form "0*x" are imputed.
            try:
                len(self.x)
            except TypeError as E:
                print(E)
                return

            self.x = np.ascontiguousarray(self.x, np.complex128)

        elif isinstance(waveform, np.ndarray):
            self.x = waveform
            self.x = np.ascontiguousarray(self.x, np.complex128)

    def normalize(self):
        """Normalize the wavefunction
        """
        try:
            x2 = self.x*np.conj(self.x)
        except FloatingPointError as e:
            if "underflow" in str(e).lower():
                x2 = [0.0 if x_i == 1e-30 else 
                      x_i*np.conj(x_i) for x_i in self.x]
            else:
                raise e
        self.x = self.x/np.sqrt(np.trapz(x2, dx=self.dx))
        self.x = np.ascontiguousarray(self.x, np.complex128)

    @property
    def p(self):
        """
        The wavefunction in the momentum basis.
        """
        return np.fft.fftshift(np.fft.fft(self.x)/(self.N/10))

    def expectation_value(self, eigenvalues, eigenstates):
        """Find the expectation value of the wavefunction
        with respect to the eigenvalues and eigenstates
        of a Hermitian Operator
        """
        try:
            prob = np.abs(np.dot(self.x, eigenstates))**2
            if (np.max(prob) != 0.):
                prob = prob/np.sum(prob)
        except FloatingPointError as E:
            return 0.
            print(E)
        return np.sum(np.dot(np.real(eigenvalues), prob))

    def expected_momentum(self):
        """Find the momentum expectation value of the
        wavefunction
        """
        F = np.fft.fft(self.x)
        prob = np.abs(F)**2
        if (np.max(prob) != 0.):
            prob = prob/np.sum(prob)
        freq = np.fft.fftfreq(self.N, d=self.dx)
        p = 2*np.pi*freq*self.hbar/self.L
        return np.dot(p, prob)

    def avg_and_std(self, eigenvalues, eigenstates):
        """Find the expectation value and the standard deviation
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

    def p_avg_and_std(self):
        """Find the expectation value and the standard deviation
        of the wavefunction with respect to the eigenvalues and
        eigenstates of the momentum operator
        """
        F = np.fft.fft(self.x)
        prob = np.abs(F)**2
        if (np.max(prob) != 0.):
            prob = prob/np.sum(prob)
        freq = np.fft.fftfreq(self.N, d=self.dx)
        p = 2*np.pi*freq*self.hbar/self.L

        expval = np.sum(np.dot(p, prob))
        expval2 = np.sum(np.dot(p**2, prob))
        sigma = np.sqrt(expval2 - expval**2)
        return (expval, sigma)

    def set_to_momentum_eigenstate(self):
        """Set the wavefunction to an allowable
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
        if (np.max(prob) != 0.):
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
        """Set the wavefunction to an eigenstate of
        any operator. Given the eigenvalues and eigenvectors
        of the operator, reset the wavefunction to the
        most probable eigenstate, then return the
        eigenvalue that corresponds to this eigenstate.
        """

        prob = np.abs(np.dot(self.x, eigenstates))**2
        if (np.max(prob) != 0.):
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

        super().__init__()

        if isinstance(Potential, np.ndarray):
            V = Potential
            V = V.astype(np.complex128)

        elif callable(Potential):
            x = np.linspace(self.x0, (self.L + self.x0), self.N)
            V = np.array([Potential(xi) for xi in x], np.complex128)

        V *= self._scale

        # Get constants
        m, hbar, e, L, N, dx, dt = self._get_constants()

        # Initialize A and B matrices
        # A = np.zeros([N, N], np.complex128, order='F')
        # B = np.zeros([N, N], np.complex128, order='F')

        # Get the unitary operator

        # t1 = perf_counter()

        self.U = _construct_U(V, m, hbar, N, dx, dt)
        self.U = np.ascontiguousarray(self.U, np.complex128)

        # self._HU = _HU

        # t2 = perf_counter()
        # print(t2 - t1)

        # Get the Hamiltonian from the unitary operator
        # and aquire the energy eigenstates.
        # self.set_energy_eigenstates()

    def __call__(self, wavefunction):
        """Call this class
        on a wavefunction to time evolve it.
        """
        wavefunction.x = _time_evolve_wavefunction(self.U, wavefunction.x)
        # np.copyto(wavefunction.x, _time_evolve_wavefunction(self.U, wavefunction.x))

    def _set_HU(self):
        """Set HU (the Hamiltonian times the unitary operator).
        Note that HU is not Hermitian.
        """
        # The Hamiltonian H is proportional to the
        # time derivative of U times its inverse

        # hbar = np.dtype(np.complex128)
        # self._HU = np.zeros([self.N, self.N], np.complex128)
        dt = np.dtype(np.complex128)
        ihbar = 1.0j*self.hbar
        dt = 2*self.dt
        # np.copyto(self._HU, _mat_diff((ihbar/dt), self.U, np.conj(self.U.T)))
        self._HU = _mat_diff((ihbar/dt), self.U, np.conj(self.U.T))
        # self._HU = _mat_diff((ihbar/dt), self.U, self.I)

    def set_energy_eigenstates(self):
        """Set the eigenstates and energy eigenvalues.
        """

        # t1 = perf_counter()

        self._set_HU()

        eigvals, eigvects = _get_eig(self._HU)
        eigvects = eigvects.T
        eigvals = np.sign(np.real(eigvals))*np.abs(eigvals)

        tmp_dict = {}
        for i in range(len(eigvals)):
            E = np.round(eigvals[i], 7)
            if E in tmp_dict:
                tmp_dict[E] = _vect_add(eigvects[i], tmp_dict[E])
            else:
                tmp_dict[E] = eigvects[i]

        eigvals, eigvects = tmp_dict.keys(), tmp_dict.values()
        self.energy_eigenvalues = np.array(list(eigvals))
        self.energy_eigenstates = np.array(list(eigvects), np.complex128).T

        # t2 = perf_counter()
        # print(t2 - t1)
