import numpy as np
from scipy.sparse import spdiags
from scipy.sparse.linalg import spsolve

# Parametry
Lx = 1.0    # Długość kanału
Ly = 1.0    # Wysokość kanału
nx = 41     # Liczba węzłów wzdłuż osi x
ny = 41     # Liczba węzłów wzdłuż osi y
nu = 0.1    # Lepkość dynamiczna
rho = 1.0   # Gęstość płynu
dt = 0.001  # Krok czasowy
nt = 500    # Liczba kroków czasowych

# Tworzenie siatki
dx = Lx / (nx - 1)
dy = Ly / (ny - 1)
x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, ny)
X, Y = np.meshgrid(x, y)

# Inicjalizacja pola prędkości
u = np.zeros((ny, nx))
v = np.zeros((ny, nx))

# Rozwiązanie równań Naviera-Stokesa
for n in range(nt):
    un = u.copy()
    vn = v.copy()

    # Równanie dla prędkości u
    u[1:-1, 1:-1] = (un[1:-1, 1:-1] -
                     dt / dx * un[1:-1, 1:-1] * (un[1:-1, 1:-1] - un[1:-1, :-2]) -
                     dt / dy * vn[1:-1, 1:-1] * (un[1:-1, 1:-1] - un[:-2, 1:-1]) +
                     nu * dt / dx**2 * (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, :-2]) +
                     nu * dt / dy**2 * (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[:-2, 1:-1]))

    # Równanie dla prędkości v
    v[1:-1, 1:-1] = (vn[1:-1, 1:-1] -
                     dt / dx * un[1:-1, 1:-1] * (vn[1:-1, 1:-1] - vn[1:-1, :-2]) -
                     dt / dy * vn[1:-1, 1:-1] * (vn[1:-1, 1:-1] - vn[:-2, 1:-1]) +
                     nu * dt / dx**2 * (vn[1:-1, 2:] - 2 * vn[1:-1, 1:-1] + vn[1:-1, :-2]) +
                     nu * dt / dy**2 * (vn[2:, 1:-1] - 2 * vn[1:-1, 1:-1] + vn[:-2, 1:-1]))

    # Warunki brzegowe dla prędkości u
    u[0, :] = 0
    u[-1, :] = 0
    u[:, 0] = 0
    u[:, -1] = 1

    # Warunki brzegowe dla prędkości v
    v[0, :] = 0
    v[-1, :] = 0
    v[:, 0] = 0
    v[:, -1] = 0

# Wyświetlanie wyników
import matplotlib.pyplot as plt
plt.contourf(X, Y, u, alpha=0.5, cmap='jet')
plt.colorbar()
plt.streamplot(X, Y, u, v, color='k')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Przepływ w kanale')
plt.show()