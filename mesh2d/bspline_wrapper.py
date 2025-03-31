import numpy as np

class BSplineWrapper:

    def __init__(self):
        pass

    def read_bspline_curve(self, filename):

        curve = np.loadtxt(filename, dtype=float, usecols=(0, 1))

        x = curve[:, 0]
        z = curve[:, 1]

        le_index = np.argmin(x)
        self.x_min = np.min(x)
        self.x_max = np.max(x) - self.x_min

        self.xl = x[:le_index+1]
        self.zl = z[:le_index+1]

        self.xu = x[le_index:]
        self.zu = z[le_index:]

        pass

    def get_all_surface(self, n_points=1000):
        beta = np.linspace(0, np.pi, n_points)
        xc = 0.5 * (1 - np.cos(beta))

        x_all, z_all = self.get_surface_from_x(xc)
        return x_all, z_all

    def get_surface_from_x(self, xc):

        xc += self.x_min
        xc *= self.x_max

        zl = np.interp(xc, self.xl[::-1], self.zl[::-1])
        zu = np.interp(xc, self.xu, self.zu)

        x_all = np.concatenate((xc[::-1], xc[1:]))
        z_all = np.concatenate((zl[::-1], zu[1:]))

        return x_all, z_all