"""
Constants.
"""


class Constants:
    """Fundamental constants,
    including mass, hbar, e ... etc

    Attributes:
    m [float]: mass
    hbar [float]: Reduced Planck constant
    e [float]: Charge
    x0 [float]: Initial position
    L [float]: Length of the box
    N [int]: Number of spatial steps
    dx [float]: Space stepsize
    dt [float]: Time stepsize
    _scale [float]: multiply the potential by a certain amount.

    """

    def __init__(self):
        """
        Initialize the constants.
        """

        self.m = 1.           # Mass
        self.hbar = 1.        # Reduced Planck constant
        self.e = 1.           # Charge

        self.x0 = -0.5           # Initial position
        self.L = 1.              # The Length of the box
        self.N = 512             # Number of spatial steps
        self.dx = self.L/self.N  # Space stepsize
        self.dt = 0.00001        # Time stepsize

        self._scale = (128/self.N)*5e5

    def _get_constants(self):
        """
        Return constants.
        """
        return self.m, self.hbar, self.e, self.L, self.N, self.dx, self.dt
