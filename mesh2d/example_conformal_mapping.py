import matplotlib.pyplot as plt

from conformal_mapping import ConformalMapping
from naca4digits_deprecated import Naca4Digits

# Number of cells in the i and j direction
NC = 32

"""
----- Initialize an airfoil geometry -----

Cambered NACA airfoil must respect the following limitations to successfully generate meshes:

Maximum camber = 4
Maximum camber position  <= 4
 
"""
airfoil = Naca4Digits(4, 4, 12)

# Creates Conformal Mapping object, specifying geometry and number of cells
cm = ConformalMapping(airfoil, NC)

# Generates all the nodes
cm.generate_mesh_nodes()

x, y = airfoil.get_all_surface(1000)


"""
The following script shows how to access the nodes x and y values. Generating figure the following way can be ressource intensive for large mesh NC > 32.
Possible to accelerate nodes access using NumPy arrays.
"""

fig, ax = plt.subplots(1, dpi=300)

for i in range(NC+1):
    for j in range(NC+1):
        # ax.plot(x, y, color='tab:blue')
        ax.scatter(cm.all_nodes[i][j][0], cm.all_nodes[i][j][1], 2, color='tab:red')

ax.set_aspect(1)
ax.set_xlim(-3, 3.5)
ax.set_ylim(-3, 3)

plt.show()

# Saves the mesh
cm.write_plot3d("mesh.xyz")

# fig, ax = plt.subplots(1, dpi=300)
# for i in range(NC+1):
#     # ax.plot(x, y, color='tab:blue')
#     # ax.scatter(cm.mesh_coord[i][0], cm.mesh_coord[i][1], 3, color='tab:red')
#     # ax.scatter(cm.quasi_coord[i][0], cm.quasi_coord[i][1], 3, color='tab:green')
#     # zeta = cm.to_zeta_plane(complex(x_nc[i], y_nc[i]))
#     # ax.scatter(zeta.real, zeta.imag, 3, color='tab:orange')
#
# ax.set_aspect(1)
# ax.set_xlim(-0.1, 1.1)
# ax.set_ylim(-0.5, 0.5)
# ax.grid(True)
#
# plt.show()