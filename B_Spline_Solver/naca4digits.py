import math

class Naca4Digits:

    def __init__(self, max_camber, max_camber_pos, max_thickness):

        self.max_camber = max_camber/100
        self.mc_pos = max_camber_pos/10
        self.max_thickness = max_thickness/100

        # NACA 4 digits coefficients
        self.a = 0.2969
        self.b = -0.1260
        self.c = -0.3516
        self.d = 0.2843
        self.e = -0.1015


        if (self.max_camber == 0 and self.mc_pos != 0):
            # TODO: raise Exception
            # raise ValueError("The maximum camber position must be set to 0 if the maximum camber is set to 0.")
            pass

        if self.max_camber == 0 and self.mc_pos == 0:
            self.symmetric = True
        else:
            self.symmetric = False

    def get_camber(self, xc):
        if self.symmetric:
            return 0.0
        else:
            if xc <= self.mc_pos:
                return self.max_camber / self.mc_pos**2 * (2*self.mc_pos*xc - xc**2)
            else:
                return self.max_camber / (1 - self.mc_pos) ** 2 * ((1 - 2 * self.mc_pos) + 2 * self.mc_pos * xc - xc ** 2)

    def get_d_camber(self, xc):
        if self.symmetric:
            return 0.0
        else:
            if xc <= self.mc_pos:
                return 2 * self.max_camber / self.mc_pos ** 2 * (self.mc_pos - xc)
            else:
                return 2 * self.max_camber / (1 - self.mc_pos) ** 2 * (self.mc_pos - xc)


    def get_thickness(self, xc):
        return self.max_thickness / 0.2 * (self.a * xc**(0.5) + self.b*xc + self.c*xc**2 + self.d*xc**3 + self.e*xc**4)


    def get_theta(self, xc):
        return math.atan(self.get_d_camber(xc))

    def get_surface(self, xc):
        if self.symmetric:
            return (xc, self.get_thickness(xc)), (xc, -self.get_thickness(xc))
        else:
            thickness = self.get_thickness(xc)
            camber = self.get_camber(xc)
            theta = self.get_theta(xc)

            xu = xc - thickness*math.sin(theta)
            xl = xc + thickness*math.sin(theta)

            yu = camber + thickness*math.cos(theta)
            yl = camber - thickness*math.cos(theta)

            return (xu, yu), (xl, yl)

    def get_upper(self, xc):
        if self.symmetric:
            return xc, self.get_thickness(xc)
        else:
            thickness = self.get_thickness(xc)
            camber = self.get_camber(xc)
            theta = self.get_theta(xc)
            xu = xc - thickness*math.sin(theta)
            yu = camber + thickness*math.cos(theta)
            return xu, yu

    def get_lower(self, xc):
        if self.symmetric:
            return xc, -self.get_thickness(xc)
        else:
            thickness = self.get_thickness(xc)
            camber = self.get_camber(xc)
            theta = self.get_theta(xc)
            xl = xc + thickness*math.sin(theta)
            yl = camber - thickness*math.cos(theta)
            return xl, yl


    def get_all_surface(self, n_points=1000):

        x_TE = 1

        x_upper = []
        y_upper = []

        x_lower = []
        y_lower = []

        xc = 0
        step = x_TE/n_points

        while (xc <= 1):
            (xu, yu), (xl, yl) = self.get_surface(xc)

            x_upper.append(xu)
            y_upper.append(yu)

            x_lower.append(xl)
            y_lower.append(yl)

            xc += step

        xs_all = list(reversed(x_lower)) + x_upper
        ys_all = list(reversed(y_lower)) + y_upper
        return xs_all, ys_all


# Create airfoil
airfoil = Naca4Digits(4,4 , 12)

# Get surface coordinates
x, y = airfoil.get_all_surface(450)

filename = f"NACA{int(airfoil.max_camber*100)}{int(airfoil.mc_pos*10)}{int(airfoil.max_thickness*100)}.dat"
with open(filename, "w") as f:
    for xi, yi in zip(x, y):
        f.write(f"{xi:.6f} {yi:.6f}\n")

print(f"Saved to {filename}")
