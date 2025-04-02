import numpy as np
import matplotlib.pyplot as plt

def read_PLOT3D_mesh(file_name):
    with open(file_name, 'r') as f:
        # Read all lines from the file
        data = f.readlines()
        
        # Extract the grid dimensions
        nx, ny = map(int, data[0].split())
        
        # Calculate the total number of points
        total_points = nx * ny
        
        # Initialize arrays for coordinates
        x = np.zeros((total_points))
        y = np.zeros((total_points))
        
        # Extract the coordinates (assuming they are in a single column, alternating X and Y)
        for i in range(total_points):
            x[i] = float(data[i+1])
            y[i] = float(data[i+1+total_points])
        
        x = x.reshape((ny, nx))
        y = y.reshape((ny, nx))
            
    return x, y
        
def write_plot3d_2d(x, y, q, mach, alpha, reyn, time, grid_filename="2D.x", solution_filename="2D.q"):
    """
    Write 2D PLOT3D formatted grid and solution files with reversed index order.

    Parameters:
        x, y: 2D NumPy arrays (nj, ni) representing grid points in x and y directions.
        q: 3D NumPy array (nj, ni, 4) representing flow variables (density, x-momentum, y-momentum, energy).
        mach: Freestream Mach number.
        alpha: Freestream angle-of-attack (in degrees).
        reyn: Freestream Reynolds number.
        time: Time value (optional, can be set to 0 if not time-dependent).
        grid_filename: Name of the output grid file.
        solution_filename: Name of the output solution file.
    """
    nj, ni = x.shape
    
    # Write grid file (2D.x)
    with open(grid_filename, 'w') as grid_file:
        grid_file.write(f"{ni} {nj}\n")  # Grid dimensions
        # Write x-coordinates (reverse the order: i first, then j)
        for j in range(nj):
            for i in range(ni):    
                grid_file.write(f"{x[j, i]:.16e}")
                grid_file.write("\n")
        # Write y-coordinates (reverse the order: i first, then j)
        for j in range(nj):    
            for i in range(ni):
                grid_file.write(f"{y[j, i]:.16e}")
                grid_file.write("\n")
    
    # Write solution file (2D.q)
    with open(solution_filename, 'w') as solution_file:
        solution_file.write(f"{ni} {nj}\n")  # Grid dimensions again
        # Write freestream conditions
        solution_file.write(f"{mach:.16e} {alpha:.16e} {reyn:.16e} {time:.16e}\n")
        # Write flow variables (density, x-momentum, y-momentum, energy)
        for n in range(4):  # Iterate over the 4 variables
            for j in range(nj):
                for i in range(ni):  # Reverse the order: i first, then j
                    solution_file.write(f"{q[j, i, n]:.16e}")
                    solution_file.write("\n")

def read_plot3d_2d(solution_filename):
    """
    Read 2D PLOT3D formatted solution file (q file).

    Parameters:
        solution_filename: Name of the solution file to read.

    Returns:
        ni, nj: Grid dimensions.
        mach, alpha, reyn, time: Freestream conditions.
        q: 3D NumPy array (nj, ni, 4) representing flow variables (density, x-momentum, y-momentum, energy).
    """
    with open(solution_filename, 'r') as solution_file:
        # Read grid dimensions
        ni, nj = map(int, solution_file.readline().split())
        
        # Read freestream conditions
        mach, alpha, reyn, time = map(float, solution_file.readline().split())

        # Initialize the q array (nj, ni, 4)
        q = np.zeros((nj, ni, 4))
        
        # Read flow variables
        for n in range(4):  # Iterate over the 4 variables (density, x-momentum, y-momentum, energy)
            for j in range(nj):
                for i in range(ni):  # Read in the reversed order: i first, then j
                    q[j, i, n] = float(solution_file.readline())

    return ni, nj, mach, alpha, reyn, time, q

def cell_to_vertex_centered_airfoil(q_cell):
    """
    Convert cell-centered data to vertex-centered data, with special handling for airfoil boundaries.

    Parameters:
        q_cell: 3D NumPy array of shape (ni-1, nj-1, 4) representing the cell-centered flow variables 
                (density, x-momentum, y-momentum, energy).
    
    Returns:
        q_vertex: 3D NumPy array of shape (ni, nj, 4) representing the vertex-centered data.
    """
    nj_cell, ni_cell, num_vars = q_cell.shape
    
    # The vertex-centered grid will be one size larger in both directions
    ni_vertex = ni_cell + 1
    nj_vertex = nj_cell + 1
    
    # Initialize an array for vertex-centered data
    q_vertex = np.zeros((nj_vertex, ni_vertex, num_vars))
    
    # Compute the average of adjacent cell-centered values for interior vertices
    for j in range(1, nj_vertex-1):
        for i in range(0, ni_vertex-1):
            for n in range(num_vars):
                q_vertex[j, i, n] = 0.25 * (q_cell[j, i, n] + q_cell[j, i-1, n] +
                                            q_cell[j-1, i-1, n] + q_cell[j-1, i, n])
    
    # Handle boundary conditions
    # Airfoil surface (assuming j=1 is the lower boundary for the airfoil)
    for i in range(ni_vertex-1):
        for n in range(num_vars):
            # Use nearest cell value for airfoil surface boundary (j = 1)
            q_vertex[0, i, n] = 0.5*(q_cell[0, i, n] + q_cell[0, i-1, n])
    
    # Far-field boundary (j = nj), copy nearest cell value or apply specific boundary condition
    q_vertex[-1, :, :] = q_vertex[-2, :, :]
    
    # Handle the sides (i = 0 and i = ni)
    q_vertex[:, -1, :] = q_vertex[:, 0, :]

    return q_vertex 

def cell_dummy_to_vertex_centered_airfoil(q_cell):
    """
    Convert cell-centered data to vertex-centered data, with special handling for dummy cells and airfoil boundaries.

    Parameters:
        q_cell: 3D NumPy array of shape (ni+1, nj+1, 4) representing the cell-centered flow variables 
                (density, x-momentum, y-momentum, energy), with dummy cells at the boundaries.
    
    Returns:
        q_vertex: 3D NumPy array of shape (ni, nj, 4) representing the vertex-centered data.
    """
    nj_cell, ni_cell, num_vars = q_cell.shape
    print(f"{q_cell.shape=}")
    
    # The vertex-centered grid will be reduced in both directions to exclude the dummy cells
    ni_vertex = ni_cell + 1 
    nj_vertex = nj_cell - 1 # Excluding one dummy cell at the start and one at the end
    
    # Initialize an array for vertex-centered data
    q_vertex = np.zeros((nj_vertex, ni_vertex, num_vars))
    
    # Compute the average of adjacent cell-centered values for interior vertices=
    for j in range(nj_vertex):
        for i in range(ni_vertex):
            for n in range(num_vars):
                q_vertex[j, i, n] = 0.25 * (q_cell[j, i%ni_cell, n] + q_cell[j, i-1, n] +
                                            q_cell[j+1, i-1, n] + q_cell[j+1, i%ni_cell, n])
    
    return q_vertex
    
if __name__ == "__main__":
    # Example usage:
    x, y = read_PLOT3D_mesh("x.10")
    nx, ny = x.shape
    q = np.random.rand(nx, ny, 4)  # Example flow variables (density, x-momentum, y-momentum, energy)
    
    density = np.linspace(0, 2e5, nx*nx)
    density = density.reshape(nx, ny)
    q[:, :, 0] = density
    
    # Freestream conditions
    mach = 0.85
    alpha = 5.0  # Angle of attack
    reyn = 1e6   # Reynolds number
    time = 0.0   # Time (steady-state)
    
    write_plot3d_2d(x[:-1, :], y[:-1, :], q[:-1, :, :], mach, alpha, reyn, time, grid_filename="output.xy", solution_filename="output.q")
    
    
    # # Create the plot
    # plt.figure(figsize=(8, 6))
    
    # # Plot the mesh lines (both horizontal and vertical)
    # for i in range(x.shape[0]):
    #     plt.plot(x[i, :], y[i, :])
    #     plt.plot(x[:, i], y[:, i])
    
    # # Customize the plot
    # plt.xlabel('X')
    # plt.ylabel('Y')
    # plt.title('Airfoil Mesh')
    # plt.grid(True)
    # plt.axis('equal')  # Maintain aspect ratio
    # plt.show()