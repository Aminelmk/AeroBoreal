import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

from post_process import *




def coefPond(x,x0):
    
    indices = np.zeros([2,len(x)])
    coef = np.zeros([2,len(x)])
    i = 0
    j = 0
    while i < len(x):
        if j > len(x0) - 1:
            for k in range(i,len(x)):
                indices[:,k] = [j-1,j]
            break
        elif x[i] < x0[j]:
            indices[:,i] = [j-1,j]
            i += 1
        else:
            j += 1
    
    k1 = 0
    while indices[0,k1] == -1:
        indices[0,k1] = 0
        coef[:,k1] = [1,0]
        k1 +=1 
        if k1 == len(x):
            break
    k2 = -1
    while indices[1,k2] == len(x0):
        indices[1,k2] = len(x0) - 1
        coef[:,k2] = [1,0]
        k2 -= 1 
        if k2 == -len(x) - 1:
            break
    
    for i in range(k1,len(x)+k2+1):
        dist = x0[int(indices[1,i])] - x0[int(indices[0,i])]
        coef[:,i] = [1-(x[i]-x0[int(indices[0,i])])/dist,1-(x0[int(indices[1,i])]-x[i])/dist]
        
    return coef, indices




def calculCp(y):
    
    # Étape 1
    inputs = []
    with open("SOLVEUR_COUPLE/input_main.txt", "r", encoding="utf-8") as fichier:
        for ligne in fichier:
            inputs.append(ligne)
        M = float(inputs[0].split()[-1])
        p_inf = float(inputs[2].split()[-1])
        T_inf = float(inputs[3].split()[-1])
        M = float(inputs[0].split()[-1])
        y0 = inputs[5].split()[2:]
        profil = inputs[6].split()[2:]
        for i in range(len(y0)):
            y0[i] = float(y0[i])
            profil[i] = profil[i].split("/")[-1]
        
        coef_prof,indices_prof = coefPond(y,y0)
    
    
    # Étape 2
    it_max = []
    for filename in os.listdir("temp"):  
        it_max.append(filename.split("_")[1])
    nx = int(filename.split("_")[3])
    ny = int(filename.split("_")[5])
    it_max = int(max(it_max))
    nomFichier = f"temp/output_{it_max}_nx_{nx}_ny_{ny}_Cl.csv"
    df = pd.read_csv(nomFichier, encoding="utf-8")
    x_point = df["x"].values[:]
    y_point = df["y"].values[:]
    sweep = abs(np.rad2deg(np.atan((x_point[int(ny/2)] - x_point[ny])/(y_point[ny] - y_point[int(ny/2)]))))
    y_moy = np.zeros(ny)
    for i in range(ny):
        y_moy[i] = (y_point[i] + y_point[i+1])/2
    alpha_e = df["alpha_e"].values[0:ny]
    
    coef_alpha,indices_alpha = coefPond(y,y_moy)
    alpha = np.zeros(len(y))
    for i in range(len(y)):
        alpha[i] = np.rad2deg(coef_alpha[0,i]*alpha_e[int(indices_alpha[0,i])] + coef_alpha[1,i]*alpha_e[int(indices_alpha[1,i])])

    
    
    # Étape 3
    M *= np.cos(np.deg2rad(sweep))
    Mach = []
    coef_M = []
    indices_M = []
    for i in range(len(y0)):
        Mach.append([])
        for filename in os.listdir(f"output_Euler/output_files_{profil[i]}"):
            Mach[i].append(filename.split("_")[2])
        
        Mach[i] = list(dict.fromkeys(Mach[i]))
        for j in range(len(Mach[i])):
            Mach[i][j] = float(Mach[i][j])
        coef_tem,indice_tem = coefPond([M],Mach[i])
        coef_M.append(coef_tem)
        indices_M.append(indice_tem)
    

    
    # Étape 4
    cp = []
    x_final = []
    for i in range(len(y)):
        cp_prof = []
        x_prof = []
        for k in range(2):
            i_prof = int(indices_prof[k,i])
            alpha1 = []
            alpha2 = []
            Mach_min = Mach[i_prof][int(indices_M[i_prof][0,0])]
            Mach_max = Mach[i_prof][int(indices_M[i_prof][1,0])]
            for filename in os.listdir(f"output_Euler/output_files_{profil[i_prof]}"):
                if float(filename.split("_")[2]) == Mach_min:
                    alpha1.append(float(filename.split("_")[4]))
                elif float(filename.split("_")[2]) == Mach_max:
                    alpha2.append(float(filename.split("_")[4]))
            alpha1.sort()
            alpha2.sort()

            coef_alpha1,indices_alpha1 = coefPond([alpha[i]],alpha1)
            coef_alpha2,indices_alpha2 = coefPond([alpha[i]],alpha2)
            alpha1_min = alpha1[int(indices_alpha1[0,0])]
            alpha1_max = alpha1[int(indices_alpha1[1,0])]
            alpha2_min = alpha2[int(indices_alpha2[0,0])]
            alpha2_max = alpha2[int(indices_alpha2[1,0])]
            
            
            
            x_coord,y_coord = read_PLOT3D_mesh(f"mesh/{profil[i_prof]}")
            x_prof.append(x_coord[0])
            _,_,_,_,_,_,q_1min = read_plot3d_2d(f"output_Euler/output_files_{profil[i_prof]}/output_Mach_{Mach_min:.2f}_alpha_{alpha1_min:.2f}_mesh_{profil[i_prof]}.q")
            cp_1min = compute_coeff(x_coord, y_coord, q_1min, Mach_min, alpha1_min, T_inf, p_inf)
            _,_,_,_,_,_,q_1max = read_plot3d_2d(f"output_Euler/output_files_{profil[i_prof]}/output_Mach_{Mach_min:.2f}_alpha_{alpha1_max:.2f}_mesh_{profil[i_prof]}.q")
            cp_1max = compute_coeff(x_coord, y_coord, q_1max, Mach_min, alpha1_max, T_inf, p_inf)
            _,_,_,_,_,_,q_2min = read_plot3d_2d(f"output_Euler/output_files_{profil[i_prof]}/output_Mach_{Mach_max:.2f}_alpha_{alpha2_min:.2f}_mesh_{profil[i_prof]}.q")
            cp_2min = compute_coeff(x_coord, y_coord, q_2min, Mach_max, alpha2_min, T_inf, p_inf)
            _,_,_,_,_,_,q_2max = read_plot3d_2d(f"output_Euler/output_files_{profil[i_prof]}/output_Mach_{Mach_max:.2f}_alpha_{alpha2_max:.2f}_mesh_{profil[i_prof]}.q")
            cp_2max = compute_coeff(x_coord, y_coord, q_2max, Mach_max, alpha2_max, T_inf, p_inf)
            
            cp_prof.append(coef_M[i_prof][0,0]*(coef_alpha1[0,0]*cp_1min + coef_alpha1[1,0]*cp_1max) + coef_M[i_prof][1,0]*(coef_alpha2[0,0]*cp_2min + coef_alpha2[1,0]*cp_2max))
            
            
        cp.append(coef_prof[0,i]*cp_prof[0] + coef_prof[1,i]*cp_prof[1])
        x_final.append(coef_prof[0,i]*x_prof[0] + coef_prof[1,i]*x_prof[1])
    

    corde_points = np.zeros(ny+1)
    LE = np.zeros(ny+1)
    for i in range(ny+1):
        corde_points[i] = abs(x_point[i] - x_point[i+nx*(ny+1)])
        LE[i] = x_point[i]
    coef_geo,indices_geo = coefPond(y,y_point)

    corde = np.zeros(len(y))
    dec = np.zeros(len(y))
    for i in range(len(y)):
        corde[i] = coef_geo[0,i]*corde_points[int(indices_geo[0,i])] + coef_geo[1,i]*corde_points[int(indices_geo[1,i])]
        dec[i] = coef_geo[0,i]*LE[int(indices_geo[0,i])] + coef_geo[1,i]*LE[int(indices_geo[1,i])]
        x_final[i] *= corde[i]
        x_final[i] += dec[i]
    

    return x_final,cp
    






# y = np.linspace(-0.7,0.7,100)
# y = [-0.7,0.3,0.5]

# x,cp = calculCp(y)


# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# for i in range(len(y)):
#     ax.plot(x[i],y[i]*np.ones(len(x[i])),cp[i])
# ax.invert_zaxis()

# ax.set_xlabel('x')
# ax.set_ylabel('y') 
# ax.set_zlabel('Cp')
# ax.set_title('Cp en fonction de x et y')
# plt.grid()
# plt.show()
    



























