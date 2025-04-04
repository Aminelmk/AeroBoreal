import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from read_PLOT3D import *
from FVM import cell

def conservative_variable_from_W(W, gamma=1.4):
    variable = W/W[0]
    rho = W[0]
    u = variable[1]
    v = variable[2]
    E = variable[3]
    p = (gamma-1)*rho*(E-(u**2+v**2)/2)
    T = p/(rho*287)
    
    return rho, u, v, E, T, p

def compute_coeff(x, y, q, Mach, alpha, T_inf, p_inf, chord=1.000):
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
    
    Fx = 0
    Fy = 0
    M = 0
    x_ref = chord/4
    y_ref = 0
    for i in range(nx - 1):
        p_mid = 0.5 * (q_airfoil[i, 5] + q_airfoil[i+1, 5])
        Fx += p_mid*airfoil_cells[i].n1[0]*airfoil_cells[i].Ds1
        Fy += p_mid*airfoil_cells[i].n1[1]*airfoil_cells[i].Ds1
        
        x_mid = 0.5*(airfoil_cells[i].x1 + airfoil_cells[i].x2)
        y_mid = 0.5*(airfoil_cells[i].y1 + airfoil_cells[i].y2)
        M += p_mid*(-(x_mid-x_ref)*airfoil_cells[i].n1[1] + (y_mid-y_ref)*airfoil_cells[i].n1[0])*airfoil_cells[i].Ds1
    
    L = Fy*np.cos(alpha) - Fx*np.sin(alpha)
    D = Fy*np.sin(alpha) + Fx*np.cos(alpha)
    
    C_L = L/(0.5*rho_inf*U_inf**2*chord)
    C_D = D/(0.5*rho_inf*U_inf**2*chord)
    C_M = M/(0.5*rho_inf*U_inf**2*chord**2)
    
    
    return cp_airfoil, C_L, C_D, C_M




