# Matplotlib is only required for running this example (not the naca4digits class)
import matplotlib.pyplot as plt

# Imports the Naca4Digits class
from naca4digits import Naca4Digits

"""
----- NACA 4 Digits airfoil -----
4 digits NACA airfoils are defined by 4 digits (3 integers)

1st digit :             maximum camber as the percentage of the cord length
2st digit :             position of said maximum camber in tenth of the cord length
3rd and 4th digit :     maximum thickness in percentage of the cord length
"""

# Create an airfoil object
airfoil = Naca4Digits(0, 0, 12)

# The class method .get_all_surface(n_points) returns a list of all the x coordinates
# and a list of all the y coordinates on the airfoil surface.
# 'n_points' defines the number of coordinates on the lists (2*n_points). Default value = 1000
x, y = airfoil.get_all_surface(1000)

# Example - Plotting an airfoil with Matplotlib
fig, ax = plt.subplots(1, dpi=300)
ax.plot(x, y, color='tab:red')

ax.set_xlabel("x/c")
ax.set_ylabel("y/c")
ax.set_title(f"NACA{int(airfoil.max_camber*100)}{int(airfoil.mc_pos*10)}{int(airfoil.max_thickness*100)} Airfoil")

ax.set_aspect(1)
ax.set_xlim(-0.1, 1.1)
ax.set_ylim(-0.5, 0.5)

plt.show()




