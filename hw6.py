import numpy as np

# 題目 1：Gaussian Elimination (使用 numpy.linalg.solve)
A1 = np.array([
    [1.19, 2.11, -100, 1],
    [14.2, -0.112, 12.2, -1],
    [0, 100, -99.9, 1],
    [15.3, 0.110, -13.1, -1]
])
b1 = np.array([1.12, 3.44, 2.15, 4.16])
x1 = np.linalg.solve(A1, b1)
print("Q1 Solution (Gaussian Elimination):", x1)

# 題目 2：矩陣反矩陣
A2 = np.array([
    [4, 1, -1, 0],
    [1, 3, -1, 0],
    [-1, -1, 6, 2],
    [0, 0, 2, 5]
])
A2_inv = np.linalg.inv(A2)
print("\nQ2 Inverse of A:\n", A2_inv)

# 題目 3：手動實作 Crout 法適用於三對角矩陣
def crout_tridiagonal_solve(A, b):
    n = len(b)
    L = np.zeros_like(A)
    U = np.identity(n)
    
    # Crout decomposition
    for i in range(n):
        L[i][i] = A[i][i] - sum(L[i][k] * U[k][i] for k in range(i))
        for j in range(i+1, n):
            if A[j][i] != 0:
                L[j][i] = A[j][i] - sum(L[j][k] * U[k][i] for k in range(i))
        for j in range(i+1, n):
            if A[i][j] != 0:
                U[i][j] = (A[i][j] - sum(L[i][k] * U[k][j] for k in range(i))) / L[i][i]

    # Forward substitution: L * y = b
    y = np.zeros(n)
    for i in range(n):
        y[i] = (b[i] - sum(L[i][j] * y[j] for j in range(i))) / L[i][i]

    # Backward substitution: U * x = y
    x = np.zeros(n)
    for i in reversed(range(n)):
        x[i] = y[i] - sum(U[i][j] * x[j] for j in range(i+1, n))
    
    return x

A3 = np.array([
    [3, -1, 0, 0],
    [-1, 3, -1, 0],
    [0, -1, 3, -1],
    [0, 0, -1, 3]
], dtype=float)
b3 = np.array([2, 3, 4, 1], dtype=float)

x3 = crout_tridiagonal_solve(A3, b3)
print("\nQ3 Solution (Crout method):", x3)
