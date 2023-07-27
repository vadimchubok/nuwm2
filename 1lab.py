import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import quad
Ux = lambda x: (np.exp(1) * np.exp(x) / (np.exp(2) + 1)) - (np.exp(1) * np.exp(-x) / (np.exp(2) + 1)) - x

a = 0
b = 1.0

p = 1
g = 1
f = -1

basic_func = 50
step = (b - a) / basic_func
x_vals = []
U = []
U_sol = []

A = np.zeros((basic_func, basic_func))
F = np.zeros(basic_func)

def basis_function(x, i):
    return x ** i * np.exp(x*i)

def basis_function_dx(x, i):
    return i * x ** (i-1) * np.exp(i*x) + i * x ** i * np.exp(i*x)

def a_ij(phi_i_dx, phi_j_dx, phi_i, phi_j):
    return quad(lambda x: p * phi_i_dx(x) * phi_j_dx(x), a, b)[0] + quad(lambda x: g * phi_i(x) * phi_j(x), a, b)[0]

def f_i(phi_i):
    return quad(lambda x: f * phi_i(x), a, b)[0]

for i in range(basic_func):
    for j in range(basic_func):
        A[i][j] = a_ij(lambda x: basis_function_dx(x, i + 1 ,) , 
                       lambda x: basis_function_dx(x, j + 1),
                       lambda x: basis_function(x, i + 1),
                       lambda x: basis_function(x, j + 1))
    F[i] = f_i(lambda x: basis_function(x, i + 1))

C = np.linalg.solve(A, F)

def solution(x):
    return sum(C[i] * basis_function(x, i + 1) for i in range(basic_func))

for i in range(basic_func):
    x_vals.append(a + step * i)
    U.append(Ux(a + step * i))
    U_sol.append(solution(a + step * i))

plt.plot(x_vals, U)
plt.plot(x_vals, U_sol)

plt.show()
