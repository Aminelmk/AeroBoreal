import numpy as np

class Naca4Digits:

    def __init__(self, max_camber, max_camber_pos, max_thickness, sharp=True):

        self.max_camber = max_camber/100
        self.mc_pos = max_camber_pos/10
        self.max_thickness = max_thickness/100
        self.make_sharp = sharp

        # NACA 4 digits coefficients
        self.a = 0.2969
        self.b = -0.1260
        self.c = -0.3516
        self.d = 0.2843
        self.e = -0.1015

        # Location of the sharp trailing edge (when thickness = 0)
        self.DEPRECATED_X_TE_SHARP = 1.0089304115
        self.X_TE_SHARP = 1.008930403914562

        if self.max_camber == 0 or self.mc_pos == 0:
            self.symmetric = True
        else:
            self.symmetric = False

    def get_thickness(self, xc):
        return self.max_thickness / 0.2 * (self.a * xc**(0.5) + self.b*xc + self.c*xc**2 + self.d*xc**3 + self.e*xc**4)

    def get_camber(self, xc):
        if self.symmetric:
            return np.zeros_like(xc)
        else:
            camber = np.where(xc <= self.mc_pos,
                              self.max_camber / self.mc_pos**2 * (2*self.mc_pos*xc - xc**2),
                              self.max_camber / (1 - self.mc_pos) ** 2 * ((1 - 2 * self.mc_pos) + 2 * self.mc_pos * xc - xc ** 2))
            return camber

    def get_d_camber(self, xc):
        """
        Returns the derivative of the camber relative to xc (x normalized by the chord)
        """
        if self.symmetric:
            return np.zeros_like(xc)
        else:
            d_camber = np.where(xc <= self.mc_pos,
                                2 * self.max_camber / self.mc_pos ** 2 * (self.mc_pos - xc),
                                2 * self.max_camber / (1 - self.mc_pos) ** 2 * (self.mc_pos - xc))
            return d_camber

    def get_theta(self, xc):
        return np.arctan(self.get_d_camber(xc))

    def get_upper(self, xc):
        if self.symmetric:
            return xc, self.get_thickness(xc)
        else:
            thickness = self.get_thickness(xc)
            camber = self.get_camber(xc)
            theta = self.get_theta(xc)
            xu = xc - thickness*np.sin(theta)
            zu = camber + thickness*np.cos(theta)
            return xu, zu

    def get_lower(self, xc):
        if self.symmetric:
            return xc, -self.get_thickness(xc)
        else:
            thickness = self.get_thickness(xc)
            camber = self.get_camber(xc)
            theta = self.get_theta(xc)
            xl = xc + thickness*np.sin(theta)
            zl = camber - thickness*np.cos(theta)
            return xl, zl


    def get_all_surface(self, n_points=1000):

        beta = np.linspace(0, np.pi, n_points)
        xc = 0.5 * (1 - np.cos(beta))

        x_all, z_all = self.get_surface_from_x(xc)
        return x_all, z_all


    def get_surface_from_x(self, xc):

        if self.make_sharp:
            xc *= self.X_TE_SHARP

        xu, zu = self.get_upper(xc)
        xl, zl = self.get_lower(xc)

        x_all = np.concatenate((xl[::-1], xu[1:]))
        z_all = np.concatenate((zl[::-1], zu[1:]))

        if self.make_sharp:
            x_all /= self.X_TE_SHARP
            z_all /= self.X_TE_SHARP

            x_te_mean = 0.5*(x_all[0] + x_all[-1])
            z_te_mean = 0.5*(z_all[0] + z_all[-1])

            x_all[0] = x_te_mean
            x_all[-1] = x_te_mean
            z_all[0] = z_te_mean
            z_all[-1] = z_te_mean

        return x_all, z_all