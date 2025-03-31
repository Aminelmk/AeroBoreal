import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.animation import FuncAnimation
import copy

class PoissonMesh:
    def __init__(self, n_nodes, xs, ys):

        self.n_nodes = n_nodes

        print(len(xs))

        self.xs = xs
        self.ys = ys

        self.a = 0
        self.b = 0
        self.c = 0
        self.d = 0

        self.p = np.empty(self.n_nodes)
        self.q = np.empty(self.n_nodes)
        self.r = np.empty(self.n_nodes)
        self.s = np.empty(self.n_nodes)

        self.R1 = 0
        self.R2 = 0
        self.R3 = 0
        self.R4 = 0

        self.alpha = 0
        self.beta = 0
        self.gamma = 0

        self.s_eta = 1

        self.x_eta = np.empty((n_nodes, 2))
        self.y_eta = np.empty((n_nodes, 2))

        self.X = np.empty((n_nodes, n_nodes))
        self.Y = np.empty_like(self.X)

        self.ff_radius = 100


    def init_grid(self):

        self.theta = np.linspace(0, -2*np.pi, self.n_nodes)
        ff_center_x = 0.5
        ff_center_y = 0
        ff_radius = self.ff_radius
        self.ff_x = ff_radius * np.cos(self.theta) + ff_center_x
        self.ff_y = ff_radius * np.sin(self.theta) + ff_center_y

        self.X[:, 0] = self.xs
        self.Y[:, 0] = self.ys

        self.X[:, -1] = self.ff_x
        self.Y[:, -1] = self.ff_y

        lambda_ = -0.2
        for i in range(self.n_nodes):

            vec_x = self.X[i, -1] - self.X[i, 0]
            vec_y = self.Y[i, -1] - self.Y[i, 0]

            for j in range(1, self.n_nodes):
                t = j / (self.n_nodes - 1)              # linear distribution
                # t = (1 - np.exp(-lambda_ * (j))) / (1 - np.exp(-lambda_ * (self.n_nodes)))
                self.X[i, j] = vec_x * t + self.X[i, 0]
                self.Y[i, j] = vec_y * t + self.Y[i, 0]

    def grid_relaxation(self, tol=1e-4, max_iter=1_000):

        # # Creates dummy vertex
        X = np.vstack((self.X[-2, :], self.X))
        X = np.vstack((X, self.X[1, :]))
        #
        Y = np.vstack((self.Y[-2, :], self.Y))
        Y = np.vstack((Y, self.Y[1, :]))

        # X = copy.deepcopy(self.X)
        # Y = copy.deepcopy(self.Y)

        alpha = np.zeros_like(X)
        gamma = np.zeros_like(X)
        beta = np.zeros_like(X)

        res = 1.0

        for iter in range(max_iter):

            if res <= tol:
                break

            # Updates dummy vertex values
            X[0, :] = X[-3, :]
            X[-1, :] = X[2, :]
            #
            Y[0, :] = Y[-3, :]
            Y[-1, :] = Y[2, :]
            #
            # X[1:-1, 0] = self.xs
            # Y[1:-1, 0] = self.ys
            #
            # X[1:-1, -1] = self.ff_x
            # Y[1:-1, -1] = self.ff_y

            X_new = copy.deepcopy(X)
            Y_new = copy.deepcopy(Y)

            # for i in range(1, X.shape[0]-1):
            #     for j in range(1, X.shape[1]-1):
            #
            #         alpha[i, j] = 0.25*((X[i, j+1] - X[i, j-1])**2 + (Y[i, j+1] - Y[i, j-1])**2)
            #         gamma[i, j] = 0.25*((X[i+1, j] - X[i-1, j])**2 + (Y[i+1, j] - Y[i-1, j])**2)
            #         beta[i, j] = 0.0625*((X[i+1, j] - X[i-1, j])*(X[i, j+1] - X[i, j-1]) + (Y[i+1, j] - Y[i-1, j])*(Y[i, j+1] - Y[i, j-1]))
            #
            #         X_new[i, j] = (alpha[i, j]*(X[i+1, j] + X[i-1, j]) + gamma[i, j]*(X[i, j+1] + X[i, j-1]) - 2*beta[i, j]*(X[i+1, j+1] - X[i-1, j+1] - X[i+1, j-1] + X[i-1, j-1])) / (2*(alpha[i, j] + gamma[i, j]) + 1e-8)
            #         Y_new[i, j] = (alpha[i, j]*(Y[i+1, j] + Y[i-1, j]) + gamma[i, j]*(Y[i, j+1] + Y[i, j-1]) - 2*beta[i, j]*(Y[i+1, j+1] - Y[i-1, j+1] - Y[i+1, j-1] + Y[i-1, j-1])) / (2*(alpha[i, j] + gamma[i, j]) + 1e-8)

            # Define slices for inner points (to avoid edges)
            i_inner = slice(1, -1)
            j_inner = slice(1, -1)

            # Compute alpha
            alpha[i_inner, j_inner] = 0.25 * (
                    (X[i_inner, 2:] - X[i_inner, :-2]) ** 2 + (Y[i_inner, 2:] - Y[i_inner, :-2]) ** 2
            )

            # Compute gamma
            gamma[i_inner, j_inner] = 0.25 * (
                    (X[2:, j_inner] - X[:-2, j_inner]) ** 2 + (Y[2:, j_inner] - Y[:-2, j_inner]) ** 2
            )

            # Compute beta
            beta[i_inner, j_inner] = 0.0625 * (
                    (X[2:, j_inner] - X[:-2, j_inner]) * (X[i_inner, 2:] - X[i_inner, :-2]) +
                    (Y[2:, j_inner] - Y[:-2, j_inner]) * (Y[i_inner, 2:] - Y[i_inner, :-2])
            )

            # Compute X_new
            X_new[i_inner, j_inner] = (
                                              alpha[i_inner, j_inner] * (X[2:, j_inner] + X[:-2, j_inner]) +
                                              gamma[i_inner, j_inner] * (X[i_inner, 2:] + X[i_inner, :-2]) -
                                              2 * beta[i_inner, j_inner] * (
                                                          X[2:, 2:] - X[:-2, 2:] - X[2:, :-2] + X[:-2, :-2])
                                      ) / (2 * (alpha[i_inner, j_inner] + gamma[i_inner, j_inner]) + 1e-10)

            # Compute Y_new
            Y_new[i_inner, j_inner] = (
                                              alpha[i_inner, j_inner] * (Y[2:, j_inner] + Y[:-2, j_inner]) +
                                              gamma[i_inner, j_inner] * (Y[i_inner, 2:] + Y[i_inner, :-2]) -
                                              2 * beta[i_inner, j_inner] * (
                                                          Y[2:, 2:] - Y[:-2, 2:] - Y[2:, :-2] + Y[:-2, :-2])
                                      ) / (2 * (alpha[i_inner, j_inner] + gamma[i_inner, j_inner]) + 1e-10)

            res = np.linalg.norm(X[1:-1, :] - X_new[1:-1, :])/np.prod(X.shape)
            res += np.linalg.norm(Y[1:-1, :] - Y_new[1:-1, :])/np.prod(Y.shape)
            print(f"iter = {iter}/{max_iter}  |  res = {res:.4e}")
            X = copy.deepcopy(X_new)
            Y = copy.deepcopy(Y_new)
            # --- end of iteration ---

        self.X = X[1:-1, :]
        self.Y = Y[1:-1, :]

    def compute_init_derivative(self):
        pass

    def get_P(self, xi, eta):
        pass

    def get_Q(self, xi, eta):
        pass

    def write_plot3d(self, filename="mesh.xyz"):

        n_xi = self.X.shape[1]
        n_eta = self.Y.shape[0]

        print(f"Writing Plot3D grid {filename} to files.")

        file = open(filename, "w")
        file.write(f"{n_xi} {n_eta}\n")

        # Writes all x coordinates
        for i in range(n_eta):
            for j in range(n_xi):
                file.write(f"{self.X[j, i]}\n")

        # Writes all y coordinates
        for i in range(n_eta):
            for j in range(n_xi):
                file.write(f"{self.Y[j, i]}\n")

        print(f"{filename} is saved.")

