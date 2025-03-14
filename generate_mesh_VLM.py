import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from write_vtu import save_vtu_mesh

def mesh_wing(ny, nx, y0, span, cord, glisse=0, sweep=0, lam=1, diedre=0, twist=0, LE_position=0, factor=1, vstab=False):
    x1 = np.zeros([nx+1,ny+1])
    y1 = np.zeros([nx+1,ny+1])
    z1 = np.zeros([nx+1,ny+1])
    xlinspace = np.linspace(0,cord,nx+1)
    #ylinspace = np.linspace(0,span,ny+1) # Distribution linéaire des y
    theta = np.linspace(np.pi/2,np.pi,ny+1)
    ylinspace = -1*np.cos(theta)*span
    #zlinspace = np.linspace(0,span*np.tan(np.deg2rad(diedre)),ny+1) # Old version qui créer une courbe au lieu d'une ligne droite
    zlinspace = (ylinspace)*np.tan(np.deg2rad(diedre))
    angletwist_linespace = -1*np.cos(theta)*twist
    angletwist_linespace[0]=0
    
    for j in range(ny+1):
        for i in range(nx+1):
            y1[i,j] = ylinspace[j]
    for i in range(nx+1):
        for j in range(ny+1):
            x1[i,j] = xlinspace[i] + y1[i,j]*np.tan(np.deg2rad(sweep)) + y1[i,j]/span*cord*(1-lam)/2*(1 - 2*i/nx)
    for j in range(ny+1):
        for i in range(nx+1):
            z1[i,j] = zlinspace[j]
    
    for j in range(ny+1) :
        corde = x1[-1, j] - x1[0, j]
        for i in range(nx+1) :
            z1[i, j] += (corde/4-x1[i,j])*np.tan(np.deg2rad(angletwist_linespace[j]))
    
    x1 *= factor
    y1 *= factor
    z1 *= factor
    
    x1 += LE_position
    
    if vstab :
        return x1, z1, y1
    
    else :
        x_left = np.flip(x1, 1)
        y_left = np.flip(-y1, 1)
        z_left = np.flip(z1, 1)
        
        x_mesh = np.concatenate((x_left[0], x1[0])).reshape((1, 2*(ny+1)))
        y_mesh = np.concatenate((y_left[0], y1[0])).reshape((1, 2*(ny+1)))
        z_mesh = np.concatenate((z_left[0], z1[0])).reshape((1, 2*(ny+1)))
        for i in range(1, nx+1) :
            x_mesh = np.vstack((x_mesh, np.concatenate((x_left[i], x1[i]))))
            y_mesh = np.vstack((y_mesh, np.concatenate((y_left[i], y1[i]))))
            z_mesh = np.vstack((z_mesh, np.concatenate((z_left[i], z1[i]))))
        
        x_mesh = np.delete(x_mesh, ny, 1) # Remove duplicate coords
        y_mesh = np.delete(y_mesh, ny, 1) # Remove duplicate coords
        z_mesh = np.delete(z_mesh, ny, 1) # Remove duplicate coords
    
    
    
    
        return x_mesh, y_mesh, z_mesh

def lovell_mesh_wing(nx, ny, LE_position, factor=1, vstab=False, y0=0, z0=0) :
    """
    Generates a wing or tailplane mesh.
    
    Parameters:
    nx (int): Number of chordwise panels.
    ny (int): Number of spanwise panels.
    LE_position (float): Leading edge x-position.
    factor (float, optional): Scaling factor. Default is 1.
    vstab (bool, optional): If True, generates a vertical stabilizer.
    
    Returns:
    Tuple of numpy arrays (x, y, z)
    """ 
    
    theta = np.linspace(np.pi/2,np.pi,ny+1)
    y_span = -1*np.cos(theta)*1.074 # Spanwise panel coordinates
    
    x_lower = 0.632887/1.074*y_span # Chordwise panel coordinates for lower surface
    x_upper = (0.76621-0.380923)/1.074*y_span + 0.380923 # Chordwise panel coordinates for upper surface
    
    # Generate the mesh
    x = np.zeros((nx+1, ny+1))
    y = np.zeros((nx+1, ny+1))
    z = np.zeros((nx+1, ny+1))
    for i in range(nx+1):
        x[i] = (x_upper - x_lower)*i/(nx) + x_lower
        y[i] = y_span
        
    x *= factor
    y *= factor
    z *= factor
    
    x += LE_position
    y += y0
    z += z0
    
    x_left = np.flip(x, 1)
    y_left = np.flip(-y, 1)
    z_left = np.flip(z, 1)
    
    x_mesh = np.concatenate((x_left[0], x[0])).reshape((1, 2*(ny+1)))
    y_mesh = np.concatenate((y_left[0], y[0])).reshape((1, 2*(ny+1)))
    z_mesh = np.concatenate((z_left[0], z[0])).reshape((1, 2*(ny+1)))
    for i in range(1, nx+1) :
        x_mesh = np.vstack((x_mesh, np.concatenate((x_left[i], x[i]))))
        y_mesh = np.vstack((y_mesh, np.concatenate((y_left[i], y[i]))))
        z_mesh = np.vstack((z_mesh, np.concatenate((z_left[i], z[i]))))
    
    if y0 == 0 :
        x_mesh = np.delete(x_mesh, ny, 1) # Remove duplicate coords
        y_mesh = np.delete(y_mesh, ny, 1) # Remove duplicate coords
        z_mesh = np.delete(z_mesh, ny, 1) # Remove duplicate coords
    
    if vstab :
        return x, z, y
    
    else :
        return x_mesh, y_mesh, z_mesh

def save_mesh(nx, ny, x, y, z, file_name) :
    """
    Saves a structured mesh to a file.
    
    Parameters:
        nx (int): Number of chordwise panels.
        ny (int): Number of spanwise panels.
        x, y, z (numpy arrays): Mesh coordinates.
        file_name (str): Name of the output file.
    """
    mesh = np.hstack((x.flatten(), y.flatten(), z.flatten()))
    with open(file_name, "w") as f:
        f.write(f"{nx+1} {ny+1}\n")  # Write the number of coordinates as the first line
        np.savetxt(f, mesh, delimiter=',')



def mesh_fuselage(fuselage_length, fuselage_height, fuselage_width, nx_fuse, nz_fuse, ny_fuse):
    """
    Generates a fuselage mesh in the form of a rectangular cross.

    Parameters:
        fuselage_length (float): Length of the fuselage.
        fuselage_height (float): Height of the fuselage.
        fuselage_width (float): Width of the fuselage.
        nx_fuse (int): Longitudinal divisions.
        nz_fuse (int): Vertical divisions.
        ny_fuse (int): Horizontal divisions.

    Returns:
        Tuple of numpy arrays (X_fuse, Y_fuse, Z_fuse)
    """
    x_fuse = np.linspace(0, fuselage_length, nx_fuse + 1)

    # Vertical cross-section (YZ plane)
    z_vertical = np.linspace(-fuselage_height / 2, fuselage_height / 2, nz_fuse + 1)
    y_vertical = np.zeros_like(z_vertical)

    # Horizontal cross-section (YZ plane)
    y_horizontal = np.linspace(-fuselage_width / 2, fuselage_width / 2, ny_fuse + 1)
    z_horizontal = np.zeros_like(y_horizontal)

    # Generate fuselage mesh
    X_fuse, Z_fuse = np.meshgrid(x_fuse, z_vertical)  # Vertical cross-section
    _, Y_fuse = np.meshgrid(x_fuse, y_horizontal)  # Horizontal cross-section

    return X_fuse.transpose(), Y_fuse.transpose(), Z_fuse.transpose()

def plot_wing(nx, ny, x, y, z, sym=True, color="tab:blue") : 
    for i in range(nx + 1):
        ax.plot(x[i], y[i], z[i], "-", color=color)  # Chordwise lines
    for i in range(ny + 1):
        ax.plot(x[:, i], y[:, i], z[:, i], "-", color=color)  # Spanwise lines
    
    if sym : 
        for i in range(nx + 1):
            ax.plot(x[i], -y[i], z[i], "-", color=color)  # Chordwise lines
        for i in range(ny + 1):
            ax.plot(x[:, i], -y[:, i], z[:, i], "-", color=color)  # Spanwise lines

############################################ Generate and Save  Lovell Meshes ############################################
# Wing mesh            
nx_wing = 2
ny_wing = 5
x_wing, y_wing, z_wing = lovell_mesh_wing(nx_wing, ny_wing, 0.7300327456656371, z0=-0.0368)
save_mesh(nx_wing, 2*ny_wing, x_wing, y_wing, z_wing, "mesh_wing.txt")
save_vtu_mesh(x_wing, y_wing, z_wing, "mesh_wing.vtu")

# Horizontal stabilizer
nx_hstab = 1
ny_hstab = 5
x_hstab, y_hstab, z_hstab = lovell_mesh_wing(nx_hstab, ny_hstab, 2.239, 0.5)
save_mesh(nx_hstab, 2*ny_hstab, x_hstab, y_hstab, z_hstab, "mesh_hstab.txt")
save_vtu_mesh(x_hstab, y_hstab, z_hstab, "mesh_hstab.vtu")

# Vertical stabilizer
nx_vstab = 1
ny_vstab = 5
x_vstab, y_vstab, z_vstab = lovell_mesh_wing(nx_vstab, ny_vstab, 2.239, 0.5, True)
save_mesh(nx_vstab, ny_vstab, x_vstab, y_vstab, z_vstab, "mesh_vstab.txt")
save_vtu_mesh(x_vstab, y_vstab, z_vstab, "mesh_vstab.vtu")

# Fuselage mesh
fuselage_length = 2.239
fuselage_height = 0.3048
fuselage_width = 0.3048
nx_fuse, ny_fuse, nz_fuse = 10, 1, 1
X_fuse, Y_fuse, Z_fuse = mesh_fuselage(fuselage_length, fuselage_height, fuselage_width, nx_fuse, nz_fuse, ny_fuse)
save_mesh(nx_fuse, ny_fuse, X_fuse, Y_fuse, np.zeros_like(X_fuse), "mesh_fuse_horizontal.txt")
save_vtu_mesh(X_fuse, Y_fuse, np.zeros_like(X_fuse), "mesh_fuse_horizontal.vtu")
save_mesh(nx_fuse, nz_fuse, X_fuse, np.zeros_like(X_fuse), Z_fuse, "mesh_fuse_vertical.txt")
save_vtu_mesh(X_fuse, np.zeros_like(X_fuse), Z_fuse, "mesh_fuse_vertical.vtu")
# ############################################ Generate and Save Meshes ############################################
# # Wing mesh            
# nx_wing = 1
# ny_wing = 40
# AR = 9 #CRM 9.0
# sweep = 0 #CRM 35 deg
# lam = 0.5
# diedre = 0
# glisse = 0
# twist = 20
# y0 = 10
# corde = 1
# span = AR*(corde+lam)/4
# x_wing, y_wing, z_wing = mesh_wing(ny_wing, nx_wing, y0, span, corde, glisse, sweep, lam, diedre, twist, 2.5)
# save_mesh(nx_wing, 2*ny_wing, x_wing, y_wing, z_wing, "mesh_wing.txt")

# # Horizontal stabilizer
# nx_hstab = 1
# ny_hstab = 5
# AR = 4 #CRM 9.0
# sweep = 0 #CRM 35 deg
# lam = 0.5
# diedre = 0
# glisse = 0
# twist = 0
# y0 = 10
# corde = 0.5
# span = AR*(corde+lam)/4
# x_hstab, y_hstab, z_hstab = mesh_wing(ny_hstab, nx_hstab, y0, span, corde, glisse, sweep, lam, diedre, twist, 10.0)
# save_mesh(nx_hstab, 2*ny_hstab, x_hstab, y_hstab, z_hstab, "mesh_hstab.txt")

# # Vertical stabilizer
# nx_vstab = 1
# ny_vstab = 5
# AR = 4 #CRM 9.0
# sweep = 0 #CRM 35 deg
# lam = 0.5
# diedre = 0
# glisse = 0
# twist = 0
# y0 = 10
# corde = 0.5
# span = AR*(corde+lam)/4
# x_vstab, y_vstab, z_vstab = mesh_wing(ny_hstab, nx_hstab, y0, span, corde, glisse, sweep, lam, diedre, twist, 10.0, 1, True)
# save_mesh(nx_vstab, ny_vstab, x_vstab, y_vstab, z_vstab, "mesh_vstab.txt")

# # Fuselage mesh
# fuselage_length = 10.0
# fuselage_height = 1.0
# fuselage_width = 1.0
# nx_fuse, ny_fuse, nz_fuse = 2, 1, 1
# X_fuse, Y_fuse, Z_fuse = mesh_fuselage(fuselage_length, fuselage_height, fuselage_width, nx_fuse, nz_fuse, ny_fuse)
# save_mesh(nx_fuse, ny_fuse, X_fuse, Y_fuse, np.zeros_like(X_fuse), "mesh_fuse_horizontal.txt")
# save_mesh(nx_fuse, nz_fuse, X_fuse, np.zeros_like(X_fuse), Z_fuse, "mesh_fuse_vertical.txt")

############################################ 3D Plot ############################################
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

plot_wing(nx_wing, 2*ny_wing, x_wing, y_wing, z_wing, False, "tab:blue")
plot_wing(nx_hstab, 2*ny_hstab, x_hstab, y_hstab, z_hstab, False, "tab:orange")
plot_wing(nx_vstab, ny_vstab, x_vstab, y_vstab, z_vstab, False, "tab:green")
plot_wing(nx_fuse, nz_fuse, X_fuse, np.zeros_like(X_fuse), Z_fuse, False, "tab:red")
plot_wing(nx_fuse, ny_fuse, X_fuse, Y_fuse, np.zeros_like(Y_fuse), False, "tab:red")


ax.set_xlim(0, fuselage_length)
ax.set_ylim(-fuselage_length/2, fuselage_length/2)
ax.set_zlim(-fuselage_length/2, fuselage_length/2)

# Labels and grid
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.grid()









