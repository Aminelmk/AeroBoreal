import functools
import sys
import math
from scipy import optimize
from scipy.optimize import bracket

import numpy as np


class ConformalMapping:

    def __init__(self, airfoil, NC):

        self.airfoil = airfoil

        # Singular points
        self.z0 = None
        self.z1 = None
        self.zeta0 = None
        self.zeta1 = None

        self.tau = None
        self.k = None

        self.rho = None

        # Trailing edge and leading edge coordinates
        self.TE_x = None
        self.TE_y = None

        self.LE_x = None
        self.LE_y = None

        # Machine epsilon
        self.eps = math.sqrt(sys.float_info.epsilon)

        # Defines number of cell in the i and j direction
        self.NC = NC


    def get_closed_TE(self):

        # TODO: should be y_surface fct of x_surface (not x_chord)
        def delta_TE(xs):
            (_, yu), (_, yl) = self.airfoil.get_surface(xs)
            return (yu - yl)**2

        # TODO: review convergence criteria
        # TODO: check convergences
        sol = optimize.root_scalar(delta_TE, x0=1.0, method='newton', xtol=self.eps, maxiter=1_000)

        xc = sol.root
        _, y = self.airfoil.get_upper(xc)

        return xc, y

    def get_tau(self):

        # TODO: because of xc and xs offset on NACA profiles, this is not accurate
        # Upper surface finite difference (second order)
        h = self.eps
        _, y0 = self.airfoil.get_upper(self.TE_x)
        _, y1 = self.airfoil.get_upper(self.TE_x - h)
        _, y2 = self.airfoil.get_upper(self.TE_x - 2*h)
        dydx_upper = (3*y0 - 4*y1 + y2)/(2*h)

        # Lower surface finite difference
        _, y0 = self.airfoil.get_lower(self.TE_x)
        _, y1 = self.airfoil.get_lower(self.TE_x - h)
        _, y2 = self.airfoil.get_lower(self.TE_x - 2*h)
        dydx_lower = (3*y0 - 4*y1 + y2)/(2*h)

        tau_upper = math.atan(dydx_upper)
        tau_lower = math.atan(dydx_lower)

        return tau_lower - tau_upper


    def get_singular_points(self):

        # Gets trailing edge coordinates
        self.TE_x, self.TE_y = self.get_closed_TE()

        # Gets trainling edge angle
        self.tau = self.get_tau()
        self.k = 2 - self.tau/math.pi

        # TODO: make compatible with other geometry
        # Gets NACA airfoil 0.25*radius
        self.rho = 1.1019*self.airfoil.max_thickness**2

        # Maps airfoil TE and rho to singular points
        self.z0 = self.TE_x
        self.z1 = 0.5*self.rho

        self.z_center = 0.5*(self.z0 + self.z1)

        self.zeta0 = (self.z0 - self.z_center) / self.k + self.z_center
        self.zeta1 = (self.z1 - self.z_center) / self.k + self.z_center

        # TODO: see how to handle TE imaginary component
        self.zeta_TE = self.to_zeta_plane(complex(self.TE_x, self.TE_y))
        self.zeta_LE = self.to_zeta_plane(complex(0, self.TE_y))

        ratio = (self.z0 - self.zeta0) / self.z0
        self.zeta_center = complex(0.5*(self.zeta_LE + self.zeta_TE), ratio*self.airfoil.max_camber)

        print(f"zeta_center = {0.5*(self.zeta_LE + self.zeta_TE)}")


    def to_physical_plane(self, zeta):

        A = pow(((zeta - self.zeta0) / (zeta - self.zeta1)), self.k)
        return (self.z0 - A*self.z1)/(1.0 - A) + complex(0, self.TE_y)

    def to_zeta_plane(self, z):
        z = z - complex(0, self.TE_y)

        A = pow(((z - self.z0) / (z - self.z1)), 1 / self.k)
        return (self.zeta0 - A * self.zeta1) / (1.0 - A)

    def get_airfoil_mapping(self):

        self.r_init = self.zeta0 - self.zeta_center.real
        step = 2 * math.pi / self.NC

        self.thetas = []
        self.radius = []
        self.mesh_coord = []
        self.quasi_coord = []

        sing_norm = complex(self.zeta0, 0) - self.zeta_center
        theta_TE = math.atan2(sing_norm.imag, sing_norm.real)

        self.thetas.append(theta_TE)
        self.radius.append(math.sqrt(sing_norm.real**2 + sing_norm.imag**2))

        self.mesh_coord.append((self.TE_x, self.TE_y))
        self.quasi_coord.append((self.zeta0.real, self.zeta0.imag))

        # sing_norm = complex(self.zeta1, 0) - self.zeta_center
        # theta_test = math.atan2(sing_norm.imag, sing_norm.real)

        sing_norm = complex(self.zeta_LE, 0) - self.zeta_center
        theta_LE = (math.atan2(sing_norm.imag, sing_norm.real) + 2*math.pi) % (2*math.pi)

        def mapping_disc(xc, theta_target):

            xs, ys = 0, 0

            if theta_target < theta_LE:
                xs, ys = self.airfoil.get_upper(xc)
            else:
                xs, ys = self.airfoil.get_lower(xc)

            zeta = self.to_zeta_plane(complex(xs, ys)) - self.zeta_center

            radius = math.sqrt(zeta.real**2 + zeta.imag**2)
            theta = (math.atan2(zeta.imag, zeta.real) + 2*math.pi) % (2*math.pi)

            return (theta - theta_target)**2


        print(f"Starting airfoil initial mapping.")

        for i in range(1, self.NC):

            theta = i*step + theta_TE
            zeta = None
            sol = None

            fun = lambda xc: mapping_disc(xc, theta)
            bounds = [(0+self.eps, self.TE_x+self.eps)]

            # TODO: raise exception if convergence fails
            # res = optimize.minimize(fun, 0.5, method='L-BFGS-B', bounds=bounds, options={'maxiter': 2_000, 'ftol':self.eps})
            res = optimize.direct(fun, bounds=bounds, f_min=0.0)

            # print(f"error = {res.fun}")
            # print(f"status = {res.success}")

            # --- Place in class method ---
            xs, ys = 0, 0

            if theta <= theta_LE:
                xs, ys = self.airfoil.get_upper(res.x)
            else:
                xs, ys = self.airfoil.get_lower(res.x)

            zeta = self.to_zeta_plane(complex(xs, ys)) - self.zeta_center

            self.radius.append(math.sqrt(zeta.real ** 2 + zeta.imag ** 2))
            self.thetas.append((math.atan2(zeta.imag, zeta.real) + 2 * math.pi) % (2 * math.pi))

            self.mesh_coord.append((xs, ys))

            zeta = zeta + self.zeta_center
            self.quasi_coord.append((zeta.real, zeta.imag))
            # ----- -----

        self.radius.append(self.radius[0])
        self.thetas.append(self.thetas[0])
        self.mesh_coord.append(self.mesh_coord[0])
        self.quasi_coord.append(self.quasi_coord[0])

        print(f"Aifoil initial mapping completed.")


    def get_quasi_radius(self):
        # TODO: this is not accurate --> need to compute quasi-circle circumference and then find radius

        circum = 0

        for i in range(self.NC):
            zeta0 = self.get_zeta_coord(self.radius[i], self.thetas[i])
            zeta1 = self.get_zeta_coord(self.radius[i+1], self.thetas[i+1])

            circum += math.sqrt((zeta1.real-zeta0.real)**2+(zeta1.imag-zeta0.imag)**2)

        return circum/(2*math.pi)

    def get_extended_quasi_radius(self, R1, j):
        return R1*math.exp(j*2*math.pi/self.NC)

    def generate_mesh_nodes(self):

        self.get_singular_points()
        self.get_airfoil_mapping()

        print("Generating all mesh nodes.")

        self.all_nodes = [[None] * (self.NC+1) for _ in range(self.NC+1)]

        R1 = self.get_quasi_radius()
        R_NC_p1 = self.get_extended_quasi_radius(R1, self.NC+1)

        for j in range(self.NC+1):

            Rj = self.get_extended_quasi_radius(R1, j)

            for i in range(self.NC+1):
                # Computes corrected radius
                ri = self.radius[i]
                radius = (ri*(R_NC_p1 - R1) + R_NC_p1*(Rj - R1))/(R_NC_p1 - R1)

                zeta = self.get_zeta_coord(radius, self.thetas[i])
                z = self.to_physical_plane(zeta)
                self.all_nodes[i][j] = (z.real, z.imag)

        print(f"All nodes were generated.")

    def get_zeta_coord(self, r, theta):
        return complex(r*math.cos(theta), r*math.sin(theta)) + complex(self.zeta_center, 0)

    def write_plot3d(self, filename="mesh.xyz"):

        print("Writing Plot3D mesh file to files.")

        file = open(filename, "w")
        file.write("1\n")
        file.write(f"{self.NC+1} {self.NC+1}\n")

        # Writes all x coordinates
        for j in range(self.NC+1):
            for i in range(self.NC+1):
                file.write(f"{self.all_nodes[i][j][0]}\n")

        # Writes all y coordinates
        for j in range(self.NC+1):
            for i in range(self.NC+1):
                file.write(f"{self.all_nodes[i][j][1]}\n")

        print("Mesh was saved to files.")

