import numpy as np
import scipy.optimize as so
import os

class ConformalMapping:

    def __init__(self, naca_airfoil):

        self.X_TE_SHARP = 1.008930403914562

        self.airfoil = naca_airfoil
        self.airfoil.make_sharp = False

        self.z_le = None
        self.z_te = None

        self.z0 = None
        self.z1 = None

        self.zeta0 = None
        self.zeta1 = None

        self.fit_airfoil()

    def fit_airfoil(self):

        self.z_te = self.X_TE_SHARP
        self.z_le = 0.0

        self.z0 = self.z_te
        self.z1 = 0.5*1.1019*self.airfoil.max_thickness**2

        step = 1e-8
        d_thickness = (3*self.airfoil.get_thickness(self.z_te) - 4*self.airfoil.get_thickness(self.z_te - 1*step) + self.airfoil.get_thickness(self.z_te - 2*step))/(2*step)
        print(f'd_thickness = {d_thickness}')

        self.tau = 2*np.arctan(abs(d_thickness))
        print(f"tau = {self.tau}")
        self.k = 2 - self.tau/np.pi
        self.z_center = 0.5*(self.z0 + self.z1)

        self.zeta0 = (self.z0 - self.z_center)/self.k + self.z_center
        self.zeta1 = (self.z1 - self.z_center)/self.k + self.z_center

        self.zeta_te = self.to_zeta_plane(self.z_te)
        self.zeta_le = self.to_zeta_plane(self.z_le)

        self.circle_center = 0.5*(self.zeta_le + self.zeta_te)
        self.init_radius = self.zeta0 - self.circle_center


    def to_physical_plane(self, zeta):
        lamb = pow(((zeta - self.zeta0) / (zeta - self.zeta1)), self.k)
        return (self.z0 - lamb*self.z1)/(1.0 - lamb)

    def to_zeta_plane(self, z):
        lamb = pow(((z - self.z0) / (z - self.z1)), 1/self.k)
        return (self.zeta0 - lamb*self.zeta1)/(1.0 - lamb)

    def generate_mesh(self, n_cell):

        n_nodes = n_cell + 1

        self.n_cell = n_cell
        self.n_nodes = n_nodes

        thetas = np.linspace(0, 2*np.pi, n_nodes, endpoint=True)
        radius = np.empty_like(thetas)
        zeta_airfoil = [None] * n_nodes
        z_airfoil = [None] * n_nodes

        self.circle_pts = np.empty((n_nodes, 2))

        for i, theta in enumerate(thetas):
            # real = self.circle_radius * np.cos(theta)
            # imag = self.circle_radius * np.sin(theta)
            #
            # print(real, imag)
            # zeta_airfoil[i] = complex(real, imag) + self.circle_center
            # z_airfoil[i] = self.to_physical_plane(zeta_airfoil[i])

            def airfoil_error(radius):
                radius = radius.item()

                # if radius < 0:
                #     radius = 0.0

                real = radius * np.cos(theta)
                imag = radius * np.sin(theta)
                zeta = complex(real, imag) + self.circle_center
                z = self.to_physical_plane(zeta)
                z_x = z.real
                z_y = z.imag
                # print(z_x, z_y)

                if z_x < 0:
                    z_x = 0.0

                if theta <= np.pi:
                    x, y = self.airfoil.get_upper(z_x)
                else:
                    x, y = self.airfoil.get_lower(z_x)
                # error = (x - z_x)**2 + (y - z_y)**2
                error = (y - z_y)**2
                # print(f"error = {error}")

                return error

            res = so.minimize(airfoil_error, self.init_radius, bounds=np.array([[self.init_radius/1.2, self.init_radius*1.2]]) ,method='Powell', options={'xtol': 1e-15, 'ftol': 1e-15, 'maxiter': 1e6})

            radius[i] = res.x[0]

            # print(f"radius = {res.x[0]}")
            print(f"error = {res.fun}")

            self.circle_pts[i, 0] = radius[i]*np.cos(theta) + self.circle_center
            self.circle_pts[i, 1] = radius[i]*np.sin(theta) + self.circle_center

        self.circle_circumference = 0.0
        for i in range(len(thetas) - 1):
            pt_i = self.circle_pts[i, :]
            pt_j = self.circle_pts[i+1, :]

            self.circle_circumference += np.linalg.norm(pt_j - pt_i)

        # print(f"circle circumference: {self.circle_circumference}")

        R1 = self.circle_circumference/(2*np.pi)

        Rj = np.empty(n_nodes)

        for j in range(len(Rj)):
            Rj[j] = R1*np.exp((j*2*np.pi/n_cell))

        # self.r_ij = np.empty((n_nodes, n_nodes))

        x_coords = np.empty((n_nodes, n_nodes))
        y_coords = np.empty((n_nodes, n_nodes))

        for i in range(n_nodes):
            for j in range(n_nodes):
                r_ij = (radius[i]*(Rj[-1] - R1) + Rj[-1]*(Rj[j] - R1))/(Rj[-1] - R1)

                real = r_ij*np.cos(thetas[i])
                imag = r_ij*np.sin(thetas[i])
                zeta = complex(real, imag) + self.circle_center
                z = self.to_physical_plane(zeta)
                x = z.real
                y = z.imag

                x_coords[i, j] = x
                y_coords[i, j] = y

        ile = int(0.5*n_cell)

        for i in range(ile):
            for j in range(n_nodes):

                ic = n_cell - i

                x_avg = 0.5*(x_coords[i, j] + x_coords[ic, j])
                y_avg = 0.5*(y_coords[i, j] - y_coords[ic, j])

                x_coords[i, j] = x_avg
                x_coords[ic, j] = x_avg

                y_coords[i, j] = y_avg
                y_coords[ic, j] = -y_avg

        self.X = x_coords[::-1, :]
        self.Y = y_coords[::-1, :]

    def write_plot3d(self, filename="mesh.xyz"):

        os.makedirs(os.path.dirname(filename), exist_ok=True)

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
