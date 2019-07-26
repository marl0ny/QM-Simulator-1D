import QM_1D_TI as QM
import QM_1D_TI_Numba as QMN
from time import perf_counter
import numpy as np

x = np.linspace(-0.5, 0.5, 512)

t1 = perf_counter()
U1 = QM.Unitary_Operator_1D(x**2)
t2 = perf_counter()
print(t2 - t1)

t1 = perf_counter()
U2 = QMN.Unitary_Operator_1D(x**2)
t2 = perf_counter()
print(t2 - t1)

def cospi(x):
    return np.cos(np.array([np.pi*xi for xi in x], np.complex128))

psi1 = QM.Wavefunction_1D(cospi)
psi2 = QMN.Wavefunction_1D(cospi)

t1 = perf_counter()
for _ in range (4000):
    U1(psi1)
t2 = perf_counter()
print(t2 - t1)

t1 = perf_counter()
for _ in range (4000):
    U2(psi2)
t2 = perf_counter()
print(t2 - t1)
