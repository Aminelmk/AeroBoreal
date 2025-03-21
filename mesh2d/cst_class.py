import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import scipy.special as sp  # For the binomial coefficient (comb)


class CstAirfoil:

    def __init__(self, n_order, N1=0.5, N2=1.0):

        self.n_order = n_order
        self.A_upper = np.ones(n_order+1)
        self.A_lower = np.ones(n_order+1)

        self.N1 = N1
        self.N2 = N2

        self.imported_airfoil = None
        self.upper_airfoil = None
        self.lower_airfoil = None


    def import_points(self, input_file):
        self.imported_airfoil = np.loadtxt(input_file, dtype=float, usecols=(0, 1))

        le_index = np.argmin(self.imported_airfoil[:, 0])

        self.lower_airfoil = self.imported_airfoil[0:le_index+1, :]
        self.upper_airfoil = self.imported_airfoil[le_index:, :]

    def compute_cst(self, x, *A):

        n = len(A) - 1
        Cx = (x**self.N1) * ((1 - x)**self.N2)

        Sx = 0.0
        for i, Ai in enumerate(A):
            B = sp.comb(n, i) * (x**i) *((1 - x)**(n - i))
            Sx += Ai*B

        return Cx * Sx


    def set_n_order(self, n_order):

        if n_order < self.n_order:
            self.A_upper = self.A_upper[:n_order+1]
            self.A_lower = self.A_lower[:n_order+1]
            self.n_order = n_order
        elif n_order > self.n_order:
            self.A_upper = np.concatenate((self.A_upper, np.ones(n_order - self.n_order)))
            self.A_lower = np.concatenate((self.A_lower, np.ones(n_order - self.n_order)))
            self.n_order = n_order

    def fit_airfoil(self):
        # ----- Upper surface -----
        self.A_upper, pcov = optimize.curve_fit(self.compute_cst, self.upper_airfoil[:, 0], self.upper_airfoil[:, 1], p0=self.A_upper)

        # ----- Lower surface -----
        self.A_lower, pcov = optimize.curve_fit(self.compute_cst, self.lower_airfoil[:, 0], self.lower_airfoil[:, 1], p0=self.A_lower)

    def get_all_surface(self, n_points=1000):

        x = np.linspace(0, 1, n_points+1, endpoint=True)
        upper_fit = self.compute_cst(x, *self.A_upper)
        lower_fit = self.compute_cst(x, *self.A_lower)

        x_all = np.concatenate((x[::-1], x[1:]), axis=0)
        y_all = np.concatenate((lower_fit[::-1], upper_fit[1:]), axis=0)

        return x_all, y_all

    def get_surface_from_x(self, x):

        upper_fit = self.compute_cst(x, *self.A_upper)
        lower_fit = self.compute_cst(x, *self.A_lower)

        x_all = np.concatenate((x[::-1], x[1:]), axis=0)
        y_all = np.concatenate((lower_fit[::-1], upper_fit[1:]), axis=0)

        return x_all, y_all


if __name__ == '__main__':

    cst = CstAirfoil(3, N1=0.5, N2=1.0)

    cst.import_points('example_airfoil_points.dat')
    cst.fit_airfoil()
    x_all, y_all = cst.get_all_surface()

    fix, ax = plt.subplots()
    ax.plot(x_all, y_all, color='tab:blue')
    plt.show()
