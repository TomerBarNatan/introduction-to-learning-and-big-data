import numpy as np

if __name__ == '__main__':
    X = np.matrix([[1, -2, 5, 4], [3, 2, 1, -5], [-10, 1, -4, 6]])
    A = X.T @ X
    eigvals, eigvecs = np.linalg.eig(A)
    argsort = np.argsort(eigvals)
    distortion = np.sum(eigvals[argsort][:2])
    print(f"distortion = {distortion}")

    U = eigvecs.T[argsort][2:].T
    print(f"U_T = {U.T}")

    print(f"reconstruct X: {X @ U @ U.T}")

    objecive = np.linalg.norm(X.T - U @ U.T @ X.T) ** 2
    print(f"minimum of objective = {objecive}")