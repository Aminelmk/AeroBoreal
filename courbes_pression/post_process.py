import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from courbes_pression.FVM import conservative_variable_from_W, cell
from courbes_pression.read_PLOT3D import read_PLOT3D_mesh, read_plot3d_2d
#from metrics_functions import load_checkpoint_cpp, compute_L2_norm

def conservative_variable_from_W(W, gamma=1.4):
    variable = W/W[0]
    rho = W[0]
    u = variable[1]
    v = variable[2]
    E = variable[3]
    p = (gamma-1)*rho*(E-(u**2+v**2)/2)
    T = p/(rho*287)
    
    return rho, u, v, E, T, p

def compute_coeff(x, y, q, Mach, alpha, T_inf, p_inf, chord=1):
    a_inf = np.sqrt(1.4 * 287 * T_inf)  # Freestream speed of sound
    U_inf = Mach * a_inf  # Freestream velocity magnitude
    rho_inf = p_inf/(T_inf*287)
    ny, nx, n = q.shape
    
    q_airfoil = np.zeros((nx, 6))
    for i in range(nx) :
        q_airfoil[i] = conservative_variable_from_W(q[0, i, :])
    
    # Cells generation
    airfoil_cells = np.zeros((nx-1), dtype=object)
    for i in range(nx-1) :
        airfoil_cells[i] = cell(x[0, i], y[0, i], x[0, i+1], y[0, i+1], x[0+1, i+1], y[0+1, i+1], x[0+1, i], y[0+1, i], 0., 0., 0., 0.)
    
    cp_airfoil = (q_airfoil[:, 5] - p_inf)/(0.5*rho_inf*U_inf**2)
      
    return cp_airfoil

if __name__ == "__main__":
    # #---------------------------------------------------------------------------------------------------------
    x, y = read_PLOT3D_mesh("x.6")
    ni, nj, mach, alpha, reyn, time, q_vertex = read_plot3d_2d(r'output_files_x6\output_Mach_0.30_alpha_2.60_mesh_x.6.q')
    
    cp_airfoil = compute_coeff(x, y, q_vertex, mach, alpha, T_inf=288, p_inf=1E5)
    
    
    # # Compare with Vassberg-Jameson
    # data = pd.read_csv("NACA0012_M05_A0.csv", header=None)
    # x_csv = data[0].values
    # cp_csv = data[1].values
    
    plt.figure()
    plt.plot(x[0], cp_airfoil, label="Pi4 32x32")
    #plt.plot(x_csv, cp_csv, "--", label="Vassberg-Jameson 4096x4096")
    plt.gca().invert_yaxis()
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("Cp")
    plt.grid()
    plt.title("NACA0012 Mach=0.5 alpha=0°")
    
    
    # # #---------------------------------------------------------------------------------------------------------
    # x, y = read_PLOT3D_mesh("x.6")
    # ni, nj, mach, alpha, reyn, time, q_vertex = read_plot3d_2d(r'C:\Users\hieun\CLionProjects\Euler\solution_file\test_M05_alpha125_x6.q')
    # # x, y = read_PLOT3D_mesh("x.7")
    # # ni, nj, mach, alpha, reyn, time, q_vertex = read_plot3d_2d('test_M05_alpha125.q')
    
    # cp_airfoil, C_L, C_D, C_M = compute_coeff(x, y, q_vertex, mach, alpha, T_inf=288, p_inf=1E5)
    # print(C_L, C_D, C_M)
    
    
    # # Compare with Vassberg-Jameson
    # data = pd.read_csv("NACA0012_M05_A125.csv", header=None)
    # x_csv_l = data[0].values
    # cp_csv_l = data[1].values
    # x_csv_u = data[2].values
    # cp_csv_u = data[3].values
    
    # plt.figure()
    # plt.plot(x[0], cp_airfoil, label="Pi4 64x64")
    # plt.plot(x_csv_l, cp_csv_l, "--", color="tab:orange", label="Vassberg-Jameson 4096x4096")
    # plt.plot(x_csv_u, cp_csv_u, "--", color="tab:orange")
    # plt.gca().invert_yaxis()
    # plt.legend()
    # plt.xlabel("x")
    # plt.ylabel("Cp")
    # plt.title("NACA0012 Mach=0.5 alpha=1.25°")
    
    # # #---------------------------------------------------------------------------------------------------------
    # x, y = read_PLOT3D_mesh("x.6")
    # ni, nj, mach, alpha, reyn, time, q_vertex = read_plot3d_2d(r'C:\Users\hieun\CLionProjects\Euler\solution_file\test_M08_alpha0_x6.q')
    
    # cp_airfoil, C_L, C_D, C_M = compute_coeff(x, y, q_vertex, mach, alpha, T_inf=288, p_inf=1E5)
    # print(C_L, C_D, C_M)
    
    
    # # Compare with Vassberg-Jameson
    # data = pd.read_csv("NACA0012_M08_A0.csv", header=None)
    # x_csv = data[0].values
    # cp_csv = data[1].values
    
    # plt.figure()
    # plt.plot(x[0], cp_airfoil, label="Pi4 64x64")
    # plt.plot(x_csv, cp_csv, "--", label="Vassberg-Jameson 4096x4096")
    # plt.gca().invert_yaxis()
    # plt.legend()
    # plt.xlabel("x")
    # plt.ylabel("Cp")
    # plt.title("NACA0012 Mach=0.8 alpha=0°")

    
    # # #---------------------------------------------------------------------------------------------------------
    
    # #%%
    # farfield = [20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200, 500, 1000]
    # CL = []
    # CD = []
    # CM = []
    # CL_vortex = []
    # CD_vortex = []
    # CM_vortex = []
    # for r in farfield :
    #     x, y = read_PLOT3D_mesh(r"\\wsl.localhost\Ubuntu\home\hitra\AER8875\mesh\farfield\naca0012_elliptic_128_r" + f"{r}.xyz")
        
    #     ni, nj, mach, alpha, reyn, time, q_vertex = read_plot3d_2d(r"\\wsl.localhost\Ubuntu\home\hitra\AER8875\Euler_solver\solution_files\vortex_test" + f"\\r{r}.q")    
    #     cp_airfoil, C_L, C_D, C_M = compute_coeff(x, y, q_vertex, mach, alpha, T_inf=300, p_inf=1E5)
    #     CL.append(C_L)
    #     CD.append(C_D)
    #     CM.append(C_M)
        
    #     ni, nj, mach, alpha, reyn, time, q_vertex = read_plot3d_2d(r"\\wsl.localhost\Ubuntu\home\hitra\AER8875\Euler_solver\solution_files\vortex_test" + f"\\r{r}_vortex.q")    
    #     cp_airfoil, C_L, C_D, C_M = compute_coeff(x, y, q_vertex, mach, alpha, T_inf=300, p_inf=1E5)
    #     CL_vortex.append(C_L)
    #     CD_vortex.append(C_D)
    #     CM_vortex.append(C_M)
        
    # plt.figure()
    # plt.plot(farfield, CL, "-o", label="without vortex correction")
    # plt.plot(farfield, CL_vortex, "-o", label="with vortex correction")
    # plt.xticks(np.arange(0, 1000+1, 100))
    # plt.xlabel("Distance to farfield")
    # plt.ylabel("$C_L$")    
    # plt.grid()
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig("vortex_Cl.svg")
    
    # plt.figure()
    # plt.plot(farfield, CD, "-o", label="without vortex correction")
    # plt.plot(farfield, CD_vortex, "-o", label="with vortex correction")
    # plt.xticks(np.arange(0, 1000+1, 100))
    # plt.xlabel("Distance to farfield")
    # plt.ylabel("$C_D$")
    # plt.grid()
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig("vortex_Cd.svg")
    
    # plt.figure()
    # plt.plot(farfield, CM, "-o", label="without vortex correction")
    # plt.plot(farfield, CM_vortex, "-o", label="with vortex correction")
    # plt.xticks(np.arange(0, 1000+1, 100))
    # plt.xlabel("Distance to farfield")
    # plt.ylabel("$C_M$")
    # plt.grid()
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig("vortex_Cm.svg")
    
    # #%%   
    # x, y = read_PLOT3D_mesh(r"C:\Users\hieun\Downloads\naca0012_64.xyz")
    # ni, nj, mach, alpha, reyn, time, q_vertex = read_plot3d_2d(r"\\wsl.localhost\Ubuntu\home\hitra\AER8875\Euler_solver\solution_files\vortex_test\grid_convergence\M08A125\output_64_vortex.q")
    # cp_airfoil, C_L, C_D, C_M = compute_coeff(x, y, q_vertex, mach, alpha, T_inf=300, p_inf=1E5)
    # print(C_L, C_D, C_M)
    
    # x, y = read_PLOT3D_mesh(r"C:\Users\hieun\Downloads\naca0012_128.xyz")
    # ni, nj, mach, alpha, reyn, time, q_vertex = read_plot3d_2d(r"\\wsl.localhost\Ubuntu\home\hitra\AER8875\Euler_solver\solution_files\vortex_test\grid_convergence\M08A125\output_128_vortex.q")
    # cp_airfoil, C_L, C_D, C_M = compute_coeff(x, y, q_vertex, mach, alpha, T_inf=300, p_inf=1E5)
    # print(C_L, C_D, C_M)
    
    # x, y = read_PLOT3D_mesh(r"C:\Users\hieun\Downloads\naca0012_256.xyz")
    # ni, nj, mach, alpha, reyn, time, q_vertex = read_plot3d_2d(r"\\wsl.localhost\Ubuntu\home\hitra\AER8875\Euler_solver\solution_files\vortex_test\grid_convergence\M08A125\output_256_vortex.q")
    # cp_airfoil, C_L, C_D, C_M = compute_coeff(x, y, q_vertex, mach, alpha, T_inf=300, p_inf=1E5)
    # print(C_L, C_D, C_M)
    
    # x, y = read_PLOT3D_mesh(r"C:\Users\hieun\Downloads\naca0012_512.xyz")
    # ni, nj, mach, alpha, reyn, time, q_vertex = read_plot3d_2d(r"\\wsl.localhost\Ubuntu\home\hitra\AER8875\Euler_solver\solution_files\vortex_test\grid_convergence\M08A125\output_512_vortex.q") 
    # cp_airfoil, C_L, C_D, C_M = compute_coeff(x, y, q_vertex, mach, alpha, T_inf=300, p_inf=1E5)
    # print(C_L, C_D, C_M)
    
    # x, y = read_PLOT3D_mesh(r"C:\Users\hieun\Downloads\naca0012_1024.xyz")
    # ni, nj, mach, alpha, reyn, time, q_vertex = read_plot3d_2d(r"\\wsl.localhost\Ubuntu\home\hitra\AER8875\Euler_solver\solution_files\vortex_test\grid_convergence\M08A125\output_1024_vortex.q") 
    # cp_airfoil, C_L, C_D, C_M = compute_coeff(x, y, q_vertex, mach, alpha, T_inf=300, p_inf=1E5)
    # print(C_L, C_D, C_M)
    
    # # x, y = read_PLOT3D_mesh(r"C:\Users\hieun\Downloads\naca0012_2048.xyz")
    # # ni, nj, mach, alpha, reyn, time, q_vertex = read_plot3d_2d(r"C:\Users\hieun\Downloads\output_2048.q")    
    # # cp_airfoil, C_L, C_D, C_M = compute_coeff(x, y, q_vertex, mach, alpha, T_inf=300, p_inf=1E5)
    # # print(C_L, C_D, C_M)
    
    # #%%   
    # x, y = read_PLOT3D_mesh(r"C:\Users\hieun\Downloads\naca0012_64.xyz")
    # ni, nj, mach, alpha, reyn, time, q_vertex = read_plot3d_2d(r"C:\Users\hieun\Downloads\output_64.q")    
    # cp_airfoil, C_L, C_D, C_M = compute_coeff(x, y, q_vertex, mach, alpha, T_inf=300, p_inf=1E5)
    # print(C_L, C_D, C_M)
    
    # x, y = read_PLOT3D_mesh(r"C:\Users\hieun\Downloads\naca0012_128.xyz")
    # ni, nj, mach, alpha, reyn, time, q_vertex = read_plot3d_2d(r"C:\Users\hieun\Downloads\output_128.q")    
    # cp_airfoil, C_L, C_D, C_M = compute_coeff(x, y, q_vertex, mach, alpha, T_inf=300, p_inf=1E5)
    # print(C_L, C_D, C_M)
    
    # x, y = read_PLOT3D_mesh(r"C:\Users\hieun\Downloads\naca0012_256.xyz")
    # ni, nj, mach, alpha, reyn, time, q_vertex = read_plot3d_2d(r"C:\Users\hieun\Downloads\output_256.q")   
    # cp_airfoil, C_L, C_D, C_M = compute_coeff(x, y, q_vertex, mach, alpha, T_inf=300, p_inf=1E5)
    # print(C_L, C_D, C_M)
    
    # x, y = read_PLOT3D_mesh(r"C:\Users\hieun\Downloads\naca0012_512.xyz")
    # ni, nj, mach, alpha, reyn, time, q_vertex = read_plot3d_2d(r"C:\Users\hieun\Downloads\output_512.q")    
    # cp_airfoil, C_L, C_D, C_M = compute_coeff(x, y, q_vertex, mach, alpha, T_inf=300, p_inf=1E5)
    # print(C_L, C_D, C_M)
    
    # x, y = read_PLOT3D_mesh(r"C:\Users\hieun\Downloads\naca0012_1024.xyz")
    # ni, nj, mach, alpha, reyn, time, q_vertex = read_plot3d_2d(r"C:\Users\hieun\Downloads\output_1024.q")    
    # cp_airfoil, C_L, C_D, C_M = compute_coeff(x, y, q_vertex, mach, alpha, T_inf=300, p_inf=1E5)
    # print(C_L, C_D, C_M)
    
    # x, y = read_PLOT3D_mesh(r"C:\Users\hieun\Downloads\naca0012_2048.xyz")
    # ni, nj, mach, alpha, reyn, time, q_vertex = read_plot3d_2d(r"C:\Users\hieun\Downloads\output_2048.q")    
    # cp_airfoil, C_L, C_D, C_M = compute_coeff(x, y, q_vertex, mach, alpha, T_inf=300, p_inf=1E5)
    # print(C_L, C_D, C_M)
    
    # #%%  
    # x, y = read_PLOT3D_mesh("x.6")
    # ni, nj, mach, alpha, reyn, time, q_vertex = read_plot3d_2d(r"\\wsl.localhost\Ubuntu\home\hitra\AER8875\Euler_solver\output_64.q")    
    # cp_airfoil, C_L, C_D, C_M = compute_coeff(x, y, q_vertex, mach, alpha, T_inf=300, p_inf=1E5)
    # print(C_L, C_D, C_M)
    
    # x, y = read_PLOT3D_mesh("x.5")
    # ni, nj, mach, alpha, reyn, time, q_vertex = read_plot3d_2d(r"\\wsl.localhost\Ubuntu\home\hitra\AER8875\Euler_solver\output_128.q")    
    # cp_airfoil, C_L, C_D, C_M = compute_coeff(x, y, q_vertex, mach, alpha, T_inf=300, p_inf=1E5)
    # print(C_L, C_D, C_M)
    
    # x, y = read_PLOT3D_mesh("x.4")
    # ni, nj, mach, alpha, reyn, time, q_vertex = read_plot3d_2d(r"\\wsl.localhost\Ubuntu\home\hitra\AER8875\Euler_solver\output_256.q")    
    # cp_airfoil, C_L, C_D, C_M = compute_coeff(x, y, q_vertex, mach, alpha, T_inf=300, p_inf=1E5)
    # print(C_L, C_D, C_M)
    
    # x, y = read_PLOT3D_mesh("x.3")
    # ni, nj, mach, alpha, reyn, time, q_vertex = read_plot3d_2d(r"\\wsl.localhost\Ubuntu\home\hitra\AER8875\Euler_solver\output_512.q")    
    # cp_airfoil, C_L, C_D, C_M = compute_coeff(x, y, q_vertex, mach, alpha, T_inf=300, p_inf=1E5)
    # print(C_L, C_D, C_M)
    
    # x, y = read_PLOT3D_mesh("x.2")
    # ni, nj, mach, alpha, reyn, time, q_vertex = read_plot3d_2d(r"\\wsl.localhost\Ubuntu\home\hitra\AER8875\Euler_solver\output_1024.q")    
    # cp_airfoil, C_L, C_D, C_M = compute_coeff(x, y, q_vertex, mach, alpha, T_inf=300, p_inf=1E5)
    # print(C_L, C_D, C_M)
    
    
    
    
    # #%%
    # x, y = read_PLOT3D_mesh("x.6")
    # nc = "65"
    # ma = "05"
    # a = "0"
    # ni, nj, mach, alpha, reyn, time, q_vertex = read_plot3d_2d(r"\\wsl.localhost\Ubuntu\home\hitra\AER8875\Euler_solver\solution_files\NC" + f"{nc}_M{ma}_alpha{a}.q")
    
    # cp_airfoil, C_L, C_D, C_M = compute_coeff(x, y, q_vertex, mach, alpha, T_inf=288, p_inf=1E5)
    # print(C_L, C_D, C_M)
    
    # # Compare with Vassberg-Jameson
    # data = pd.read_csv("NACA0012_M05_A0.csv", header=None)
    # x_csv_l = data[0].values
    # cp_csv_l = data[1].values
    # # x_csv_u = data[2].values
    # # cp_csv_u = data[3].values
    
    # plt.figure()
    # plt.plot(x[0], cp_airfoil, label="Solveur Euler 512x512")
    # plt.plot(x_csv_l, cp_csv_l, "--", color="tab:orange", label="Vassberg-Jameson 4096x4096")
    # # plt.plot(x_csv_u, cp_csv_u, "--", color="tab:orange")
    # plt.gca().invert_yaxis()
    # plt.legend()
    # plt.xlabel("x")
    # plt.ylabel("Cp")
    # plt.title("NACA0012 Mach=0.5 alpha=0°")
    # plt.tight_layout()
    
    # #%%
    # x, y = read_PLOT3D_mesh("x.3")
    # nc = "513"
    # ma = "05"
    # a = "125"
    # ni, nj, mach, alpha, reyn, time, q_vertex = read_plot3d_2d(r"\\wsl.localhost\Ubuntu\home\hitra\AER8875\Euler_solver\solution_files\NC" + f"{nc}_M{ma}_alpha{a}.q")
    
    # cp_airfoil, C_L, C_D, C_M = compute_coeff(x, y, q_vertex, mach, alpha, T_inf=288, p_inf=1E5)
    # print(C_L, C_D, C_M)
    
    # # Compare with Vassberg-Jameson
    # data = pd.read_csv("NACA0012_M05_A125.csv", header=None)
    # x_csv_l = data[0].values
    # cp_csv_l = data[1].values
    # x_csv_u = data[2].values
    # cp_csv_u = data[3].values
    
    # plt.figure()
    # plt.plot(x[0], cp_airfoil, label="Solveur Euler 512x512")
    # plt.plot(x_csv_l, cp_csv_l, "--", color="tab:orange", label="Vassberg-Jameson 4096x4096")
    # plt.plot(x_csv_u, cp_csv_u, "--", color="tab:orange")
    # plt.gca().invert_yaxis()
    # plt.legend()
    # plt.xlabel("x")
    # plt.ylabel("Cp")
    # plt.title("NACA0012 Mach=0.5 alpha=1.25°")
    # plt.tight_layout()
    
    # #%%
    # nc = "513"
    # ma = "08"
    # a = "0"
    # ni, nj, mach, alpha, reyn, time, q_vertex = read_plot3d_2d(r"\\wsl.localhost\Ubuntu\home\hitra\AER8875\NC" + f"{nc}_M{ma}_alpha{a}.q")
    
    # cp_airfoil, C_L, C_D, C_M = compute_coeff(x, y, q_vertex, mach, alpha, T_inf=288, p_inf=1E5)
    # print(C_L, C_D, C_M)
    
    # # Compare with Vassberg-Jameson
    # data = pd.read_csv("NACA0012_M08_A0.csv", header=None)
    # x_csv_l = data[0].values
    # cp_csv_l = data[1].values
    # # x_csv_u = data[2].values
    # # cp_csv_u = data[3].values
    
    # plt.figure()
    # plt.plot(x[0], cp_airfoil, label="Solveur Euler 512x512")
    # plt.plot(x_csv_l, cp_csv_l, "--", color="tab:orange", label="Vassberg-Jameson 4096x4096")
    # # plt.plot(x_csv_u, cp_csv_u, "--", color="tab:orange")
    # plt.gca().invert_yaxis()
    # plt.legend()
    # plt.xlabel("x")
    # plt.ylabel("Cp")
    # plt.title("NACA0012 Mach=0.8 alpha=0°")
    # plt.tight_layout()
    
    # #%%
    # x, y = read_PLOT3D_mesh("x.4")
    # nc = "257"
    # ma = "08"
    # a = "125"
    # ni, nj, mach, alpha, reyn, time, q_vertex = read_plot3d_2d(r"\\wsl.localhost\Ubuntu\home\hitra\AER8875\NC" + f"{nc}_M{ma}_alpha{a}.q")
    
    # cp_airfoil, C_L, C_D, C_M = compute_coeff(x, y, q_vertex, mach, alpha, T_inf=288, p_inf=1E5)
    # print(C_L, C_D, C_M)
    
    # # Compare with Vassberg-Jameson
    # data = pd.read_csv("NACA0012_M08_A125.csv", header=None)
    # x_csv_l = data[0].values
    # cp_csv_l = data[1].values
    # x_csv_u = data[2].values
    # cp_csv_u = data[3].values
    
    # plt.figure()
    # plt.plot(x[0], cp_airfoil, label="Solveur Euler 512x512")
    # plt.plot(x_csv_l, cp_csv_l, "--", color="tab:orange", label="Vassberg-Jameson 4096x4096")
    # plt.plot(x_csv_u, cp_csv_u, "--", color="tab:orange")
    # plt.gca().invert_yaxis()
    # plt.legend()
    # plt.xlabel("x")
    # plt.ylabel("Cp")
    # plt.title("NACA0012 Mach=0.8 alpha=1.25°")
    # plt.tight_layout()
    
    # #---------------------------------------------------------------------------------------------------------
    # #%%
    # NC = ["33", "65", "129", "257", "513"]
    # MACH = ["05", "08"]
    # ALPHA = ["0", "125"]
    # CL = np.zeros((5, 2, 2))
    # CD = np.zeros((5, 2, 2))
    # CM = np.zeros((5, 2, 2))
    # NI = np.zeros(5)
    # for k, ma in enumerate(MACH) :
    #     for i, nc in enumerate(NC) :
    #         for j, a in enumerate(ALPHA) :
    #             x, y = read_PLOT3D_mesh(r"\\wsl.localhost\Ubuntu\home\hitra\AER8875\Euler\mesh\mesh_vassberg_jameson\vassberg_naca0012_"+f"{nc}_{nc}.x")
    #             ni, nj, mach, alpha, reyn, time, q_vertex = read_plot3d_2d(r"\\wsl.localhost\Ubuntu\home\hitra\AER8875\NC" + f"{nc}_M{ma}_alpha{a}.q")
    #             cp_airfoil, C_L, C_D, C_M = compute_coeff(x, y, q_vertex, mach, alpha, T_inf=300, p_inf=1E5)
    #             CL[i, j, k] = C_L
    #             CD[i, j, k] = C_D
    #             CM[i, j, k] = C_M
    #             NI[i] = ni
    #             print(f"{ni=}, {mach=}, {alpha=} : C_L={C_L:.9f}, C_D={C_D:.9f}, C_M={C_M:.9f}\n")
    # #%%
                
    
    # #%%
    
    


























