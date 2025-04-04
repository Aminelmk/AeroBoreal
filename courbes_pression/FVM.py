import numpy as np 
import matplotlib.pyplot as plt
from read_PLOT3D import read_PLOT3D_mesh, write_plot3d_2d, cell_to_vertex_centered_airfoil, cell_dummy_to_vertex_centered_airfoil
#from metrics_functions import compute_L2_norm, update_subplot, save_checkpoint, load_checkpoint

def conservative_variable_from_W(W, gamma=1.4):
    variable = W/W[0]
    rho = W[0]
    u = variable[1]
    v = variable[2]
    E = variable[3]
    p = (gamma-1)*rho*(E-(u**2+v**2)/2)
    T = p/(rho*287)
    
    return rho, u, v, E, T, p

class cell :
    def __init__(self, x1, y1, x2, y2, x3, y3, x4, y4, rho, u, v, E) :
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.x3 = x3
        self.y3 = y3
        self.x4 = x4
        self.y4 = y4
        
        self.rho = rho
        self.u = u
        self.v = v
        self.E = E
        
        
        self.OMEGA = 0.5*((x1-x3)*(y2-y4) + (x4-x2)*(y1-y3))
        
        self.s1 = np.array([self.y2-self.y1, self.x1-self.x2])
        self.s2 = np.array([self.y3-self.y2, self.x2-self.x3])
        self.s3 = np.array([self.y4-self.y3, self.x3-self.x4])
        self.s4 = np.array([self.y1-self.y4, self.x4-self.x1])
        
        self.Ds1 = np.linalg.norm(self.s1)
        self.Ds2 = np.linalg.norm(self.s2)
        self.Ds3 = np.linalg.norm(self.s3)
        self.Ds4 = np.linalg.norm(self.s4)
        
        self.n1 = self.s1/self.Ds1
        self.n2 = self.s2/self.Ds2
        self.n3 = self.s3/self.Ds3
        self.n4 = self.s4/self.Ds4
        
        self.W = np.array([rho, rho*u, rho*v, rho*E])
        
        self.FcDS_1 = np.array(4)
        self.FcDS_2 = np.array(4)
        self.FcDS_3 = np.array(4)
        self.FcDS_4 = np.array(4)
        
        self.Lambda_1_I = 0.
        self.Lambda_1_J = 0.
        self.Lambda_2_I = 0.
        self.Lambda_2_J = 0.
        self.Lambda_3_I = 0.
        self.Lambda_3_J = 0.
        self.Lambda_4_I = 0.
        self.Lambda_4_J = 0.
        
        self.Lambda_1_S = 0.
        self.Lambda_2_S = 0.
        self.Lambda_3_S = 0.
        self.Lambda_4_S = 0.
        
        self.eps2_2, self.eps4_2 = 0., 0.
        self.eps2_3, self.eps4_3 = 0., 0.
        self.eps2_4, self.eps4_4 = 0., 0.
        self.eps2_1, self.eps4_1 = 0., 0.

        
        self.D_1 = 0.
        self.D_2 = 0.
        self.D_3 = 0.
        self.D_4 = 0.
        
        self.R = np.array(4)
        
        
        

class spatial_discretization :
    def __init__(self, x, y, rho, u, v, E, T, p) :
        self.x = x
        self.y = y
        self.rho = rho
        self.u = u
        self.v = v
        self.E = E
        self.T = T
        self.p = p
        self.ny, self.nx = x.shape
        
         
        
        # Cells generation
        self.domain_cells = np.zeros((self.ny-1, self.nx-1), dtype=object)
        for j in range(self.ny-1) :
            for i in range(self.nx-1) :
                self.domain_cells[j,i] = cell(x[j, i], y[j, i], x[j, i+1], y[j, i+1], x[j+1, i+1], y[j+1, i+1], x[j+1, i], y[j+1, i], rho, u, v, E)
        
        # plt.figure()
        # plt.axis('equal')
        # for j in range(1) :
        #     for i in range(self.nx-1) :
        #         current_cell = self.domain_cells[j,i]
        #         x1, x2, x3, x4 = current_cell.x1, current_cell.x2, current_cell.x3, current_cell.x4
        #         y1, y2, y3, y4 = current_cell.y1, current_cell.y2, current_cell.y3, current_cell.y4
        #         n1x, n1y = current_cell.n1
        #         n2x, n2y = current_cell.n2
        #         n3x, n3y = current_cell.n3
        #         n4x, n4y = current_cell.n4
        #         plt.plot([x1, x2, x3, x4], [y1, y2, y3, y4])
        #         plt.quiver(x1+(x2-x1)/2, y1+(y2-y1)/2, n1x, n1y, angles='xy', scale_units='xy', scale=20, width=0.003)
        #         plt.quiver(x2+(x3-x2)/2, y2+(y3-y2)/2, n2x, n2y, angles='xy', scale_units='xy', scale=20, width=0.003)
        #         plt.quiver(x3+(x4-x3)/2, y3+(y4-y3)/2, n3x, n3y, angles='xy', scale_units='xy', scale=20, width=0.003)
        #         plt.quiver(x4+(x1-x4)/2, y4+(y1-y4)/2, n4x, n4y, angles='xy', scale_units='xy', scale=20, width=0.003)
        
        # Dummy cells for solid wall at airfoil
        self.solid_wall_cells = np.zeros((2, self.nx-1), dtype=object)
        # Dummy cells for farfield TO MODIFY !!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        self.farfield_cells = np.zeros((2, self.nx-1), dtype=object)
        
        self.cells = np.vstack((self.solid_wall_cells, self.domain_cells, self.farfield_cells))
    
    def compute_dummy_cells(self) :
        
        # Solid wall
        for i in range(self.nx-1) :
            rho, u, v, E, T, p2 = conservative_variable_from_W(self.cells[2, i].W)
            _, _, _, _, _, p3 = conservative_variable_from_W(self.cells[3, i].W)
            _, _, _, _, _, p4 = conservative_variable_from_W(self.cells[4, i].W)
            pw = 1/8*(15*p2 - 10*p3+ 3*p4) #Blazek
            # print(pw)
            p1 = 2*pw -p2
            vel = np.array([u, v])
            
            n = self.cells[2, i].n1
            # print(n)
            # print(pw*n)
            
            R = np.array([[-n[1], n[0]], [n[0], n[1]]])
            q_t, q_n = -np.matmul(R, vel)
            
            y_eta, x_eta = self.cells[2, i].s1/self.cells[2, i].Ds1
            
            # Swanson turkel
            u_dummy = x_eta*q_t + y_eta*q_n
            v_dummy = -y_eta*q_t + x_eta*q_n
            E = p1/(1.4-1)/rho + 0.5*(u_dummy**2 + v_dummy**2)
            
            # print(f"vel = {vel}")
            # print(f"n = {n}")
            # print(f"t = {t}")
            # print(f"qt, qn = {q_t, q_n}")
            # print(f"y_eta, x_eta = {y_eta, x_eta}")
            # print(f"udum, vdum = {u_dummy, v_dummy}")
            # current_cell = self.cells[2,i]
            # x1, x2, x3, x4 = current_cell.x1, current_cell.x2, current_cell.x3, current_cell.x4
            # y1, y2, y3, y4 = current_cell.y1, current_cell.y2, current_cell.y3, current_cell.y4
            # plt.quiver(x1+(x2-x1)/2, y1+(y2-y1)/2, u_dummy, v_dummy, angles='xy', scale_units='xy', scale=20000, width=0.003)
            # plt.quiver(x1+(x2-x1)/2, y1+(y2-y1)/2, u, v, angles='xy', scale_units='xy', scale=20000, width=0.003, color="red")
            
            self.cells[0, i] =  cell(x[0, i], y[0, i], x[0, i+1], y[0, i+1], x[1, i+1], y[1, i+1], x[1, i], y[1, i], rho, u_dummy, v_dummy, E)
            self.cells[1, i] =  cell(x[0, i], y[0, i], x[0, i+1], y[0, i+1], x[1, i+1], y[1, i+1], x[1, i], y[1, i], rho, u_dummy, v_dummy, E)
            
            
            

        # Farfield
        for i in range(self.nx-1) :
            rho, u, v, E, T, p = conservative_variable_from_W(self.cells[-3, i].W)
            c = np.sqrt(1.4*287*T)
            M = np.sqrt(u**2+v**2)/c
            nx, ny = self.cells[-3, i].n3
            if (u*nx+v*ny) > 0 : #On sort de la cellule
                if M >= 1 :
                    self.cells[-1, i] =  cell(x[-2, i], y[-2, i], x[-2, i+1], y[-2, i+1], x[-1, i+1], y[-1, i+1], x[-1, i], y[-1, i], rho, u, v, E)
                    self.cells[-2, i] =  cell(x[-2, i], y[-2, i], x[-2, i+1], y[-2, i+1], x[-1, i+1], y[-1, i+1], x[-1, i], y[-1, i], rho, u, v, E)
                else :
                    p_b = self.p
                    rho_b = rho + (p_b - p)/c**2
                    u_b = u + nx*(p - p_b)/(rho*c)
                    v_b = v + ny*(p - p_b)/(rho*c)
                    E_b = p_b/((1.4-1)*rho_b) + 0.5*(u_b**2 + v_b**2)
                    
                    W_b = np.array([rho_b, rho_b*u_b, rho_b*v_b, rho_b*E_b])
                    
                    W_a = 2*W_b - self.cells[-3, i].W
                    rho_a, u_a, v_a, E_a, T_a, p_a = conservative_variable_from_W(W_a)
                    
                    self.cells[-1, i] =  cell(x[-2, i], y[-2, i], x[-2, i+1], y[-2, i+1], x[-1, i+1], y[-1, i+1], x[-1, i], y[-1, i], rho_a, u_a, v_a, E_a)
                    self.cells[-2, i] =  cell(x[-2, i], y[-2, i], x[-2, i+1], y[-2, i+1], x[-1, i+1], y[-1, i+1], x[-1, i], y[-1, i], rho_a, u_a, v_a, E_a)
                    
            else : #On rentre de la cellule
                if M >= 1 :
                    self.cells[-1, i] =  cell(x[-2, i], y[-2, i], x[-2, i+1], y[-2, i+1], x[-1, i+1], y[-1, i+1], x[-1, i], y[-1, i], self.rho, self.u, self.v, self.E)
                    self.cells[-2, i] =  cell(x[-2, i], y[-2, i], x[-2, i+1], y[-2, i+1], x[-1, i+1], y[-1, i+1], x[-1, i], y[-1, i], self.rho, self.u, self.v, self.E)
                else :
                    p_b = 0.5*(self.p + p - rho*c*(nx*(self.u-u) + ny*(self.v-v)))
                    rho_b = self.rho + (p_b - self.p)/c**2
                    u_b = self.u - nx*(self.p - p_b)/(rho*c)
                    v_b = self.v - ny*(self.p - p_b)/(rho*c)
                    E_b = p_b/((1.4-1)*rho_b) + 0.5*(u_b**2 + v_b**2)
                    
                    W_b = np.array([rho_b, rho_b*u_b, rho_b*v_b, rho_b*E_b])
                    
                    W_a = 2*W_b - self.cells[-3, i].W
                    rho_a, u_a, v_a, E_a, T_a, p_a = conservative_variable_from_W(W_a)
                    
                    self.cells[-1, i] =  cell(x[-2, i], y[-2, i], x[-2, i+1], y[-2, i+1], x[-1, i+1], y[-1, i+1], x[-1, i], y[-1, i], rho_a, u_a, v_a, E_a)
                    self.cells[-2, i] =  cell(x[-2, i], y[-2, i], x[-2, i+1], y[-2, i+1], x[-1, i+1], y[-1, i+1], x[-1, i], y[-1, i], rho_a, u_a, v_a, E_a)
                    
                    
        
        
    def Fc(self, W, n) :
        rho, u, v, E, T, p = conservative_variable_from_W(W)
        nx, ny = n[0], n[1]
        V = nx*u + ny*v
        H = E + p/rho
        
        return np.array([rho*V, rho*u*V + nx*p,  rho*v*V + ny*p, rho*H*V])
    
    def Lamdac(self, W, n, Ds) :
        rho, u, v, E, T, p = conservative_variable_from_W(W)
        c = np.sqrt(1.4*287*T)
        nx, ny = n[0], n[1]
        V = nx*u + ny*v
        
        lamda = (abs(V) + c)*Ds
        
        return lamda
    
    def compute_Fc_DeltaS_Lamdac(self) :
        ny, nx = self.cells.shape
        for j in range(2, ny-2) :
            for i in range(nx) :
                Fc_1 = self.Fc(0.5*(self.cells[j, i].W + self.cells[j-1, i].W), self.cells[j, i].n1)
                # if j == 2 :
                #     print(f"Wji : {self.cells[j, i].W}")
                #     print(f"Wji : {conservative_variable_from_W(self.cells[j, i].W)}")
                #     print(f"Wj-1i : {conservative_variable_from_W(self.cells[j-1, i].W)}")
                #     print(f"1/2 Wji + Wj-1i : {0.5*(self.cells[j, i].W + self.cells[j-1, i].W)}")
                #     print(f"1/2 Wji + Wj-1i : {conservative_variable_from_W(0.5*(self.cells[j, i].W + self.cells[j-1, i].W))}")
                #     print(f"fc1 de {i}: {Fc_1}\n")
                Fc_2 = self.Fc(0.5*(self.cells[j, i].W + self.cells[j, (i+1)%nx].W), self.cells[j, i].n2)
                Fc_3 = self.Fc(0.5*(self.cells[j, i].W + self.cells[j+1, i].W), self.cells[j, i].n3)
                Fc_4 = self.Fc(0.5*(self.cells[j, i].W + self.cells[j, i-1].W), self.cells[j, i].n4)
                
                self.cells[j, i].FcDS_1 = Fc_1*self.cells[j, i].Ds1
                self.cells[j, i].FcDS_2 = Fc_2*self.cells[j, i].Ds2
                self.cells[j, i].FcDS_3 = Fc_3*self.cells[j, i].Ds3
                self.cells[j, i].FcDS_4 = Fc_4*self.cells[j, i].Ds4
                
        for j in range(ny) :
            for i in range(nx) :
                
                Lamda_1_I = self.Lamdac(self.cells[j, i].W, 0.5*(self.cells[j, i].n1-self.cells[j, i].n3), 0.5*(self.cells[j, i].Ds2+self.cells[j, i].Ds4))
                Lamda_1_J = self.Lamdac(self.cells[j, i].W, 0.5*(self.cells[j, i].n1-self.cells[j, i].n3), 0.5*(self.cells[j, i].Ds1+self.cells[j, i].Ds3))
                Lamda_2_I = self.Lamdac(self.cells[j, i].W, 0.5*(self.cells[j, i].n2-self.cells[j, i].n4), 0.5*(self.cells[j, i].Ds2+self.cells[j, i].Ds4))
                Lamda_2_J = self.Lamdac(self.cells[j, i].W, 0.5*(self.cells[j, i].n2-self.cells[j, i].n4), 0.5*(self.cells[j, i].Ds1+self.cells[j, i].Ds3))
                Lamda_3_I = self.Lamdac(self.cells[j, i].W, 0.5*(self.cells[j, i].n3-self.cells[j, i].n1), 0.5*(self.cells[j, i].Ds2+self.cells[j, i].Ds4))
                Lamda_3_J = self.Lamdac(self.cells[j, i].W, 0.5*(self.cells[j, i].n3-self.cells[j, i].n1), 0.5*(self.cells[j, i].Ds1+self.cells[j, i].Ds3))
                Lamda_4_I = self.Lamdac(self.cells[j, i].W, 0.5*(self.cells[j, i].n4-self.cells[j, i].n2), 0.5*(self.cells[j, i].Ds2+self.cells[j, i].Ds4))
                Lamda_4_J = self.Lamdac(self.cells[j, i].W, 0.5*(self.cells[j, i].n4-self.cells[j, i].n2), 0.5*(self.cells[j, i].Ds1+self.cells[j, i].Ds3))
                
                self.cells[j, i].Lambda_1_I = Lamda_1_I
                self.cells[j, i].Lambda_1_J = Lamda_1_J
                self.cells[j, i].Lambda_2_I = Lamda_2_I
                self.cells[j, i].Lambda_2_J = Lamda_2_J
                self.cells[j, i].Lambda_3_I = Lamda_3_I
                self.cells[j, i].Lambda_3_J = Lamda_3_J
                self.cells[j, i].Lambda_4_I = Lamda_4_I
                self.cells[j, i].Lambda_4_J = Lamda_4_J
    
    def compute_epsilon(self, cell_Im1, cell_I, cell_Ip1, cell_Ip2, k2=0.5, k4=1/64) :
        p_Im1 = conservative_variable_from_W(cell_Im1.W)[-1]
        p_I = conservative_variable_from_W(cell_I.W)[-1]
        p_Ip1 = conservative_variable_from_W(cell_Ip1.W)[-1]
        p_Ip2 = conservative_variable_from_W(cell_Ip2.W)[-1]
        
        Gamma_I = abs(p_Ip1 - 2*p_I + p_Im1)/(p_Ip1 + 2*p_I + p_Im1)
        Gamma_Ip1 = abs(p_Ip2 - 2*p_Ip1 + p_I)/(p_Ip2 + 2*p_Ip1 + p_I)
        
        eps2 = k2*max(Gamma_I, Gamma_Ip1)
        eps4 = max(0, k4-eps2)
        
        return eps2, eps4
        
        
    def compute_dissipation(self) :
        ny, nx = self.cells.shape
        
        for j in range(2, ny-2) :
            for i in range(nx) :
                cell_IJ = self.cells[j, i]
                
                cell_Ip1J = self.cells[j, (i+1)%nx]
                cell_IJp1 = self.cells[j+1, i]
                cell_Im1J = self.cells[j, i-1]
                cell_IJm1 = self.cells[j-1, i]
                
                cell_Ip2J = self.cells[j, (i+2)%nx]
                cell_IJp2 = self.cells[j+2, i]
                cell_Im2J = self.cells[j, i-2]
                cell_IJm2 = self.cells[j-2, i]
                
                Lambda_2_I = 0.5*(cell_IJ.Lambda_2_I + cell_Ip1J.Lambda_2_I)
                Lambda_2_J = 0.5*(cell_IJ.Lambda_2_J + cell_Ip1J.Lambda_2_J)
                Lambda_2_S = Lambda_2_I + Lambda_2_J
                
                Lambda_3_J = 0.5*(cell_IJ.Lambda_3_J + cell_IJp1.Lambda_3_J)
                Lambda_3_I = 0.5*(cell_IJ.Lambda_3_I + cell_IJp1.Lambda_3_I)
                Lambda_3_S = Lambda_3_I + Lambda_3_J
                
                Lambda_4_I = 0.5*(cell_IJ.Lambda_4_I + cell_Im1J.Lambda_4_I)
                Lambda_4_J = 0.5*(cell_IJ.Lambda_4_J + cell_Im1J.Lambda_4_J)
                Lambda_4_S = Lambda_4_I + Lambda_4_J
                
                Lambda_1_J = 0.5*(cell_IJ.Lambda_1_J + cell_IJm1.Lambda_1_J)
                Lambda_1_I = 0.5*(cell_IJ.Lambda_1_I+ cell_IJm1.Lambda_1_I)
                Lambda_1_S = Lambda_1_I + Lambda_1_J
                
                cell_IJ.Lambda_1_S = Lambda_1_S
                cell_IJ.Lambda_2_S = Lambda_2_S
                cell_IJ.Lambda_3_S = Lambda_3_S
                cell_IJ.Lambda_4_S = Lambda_4_S
                
                eps2_2, eps4_2 = self.compute_epsilon(cell_Im1J, cell_IJ, cell_Ip1J, cell_Ip2J)
                eps2_3, eps4_3 = self.compute_epsilon(cell_IJm1, cell_IJ, cell_IJp1, cell_IJp2)
                eps2_4, eps4_4 = self.compute_epsilon(cell_Ip1J, cell_IJ, cell_Im1J, cell_Im2J)
                eps2_1, eps4_1 = self.compute_epsilon(cell_IJp1, cell_IJ, cell_IJm1, cell_IJm2)
                
                cell_IJ.eps2_2, cell_IJ.eps4_2 = eps2_2, eps4_2
                cell_IJ.eps2_3, cell_IJ.eps4_3 = eps2_3, eps4_3
                cell_IJ.eps2_4, cell_IJ.eps4_4 = eps2_4, eps4_4
                cell_IJ.eps2_1, cell_IJ.eps4_1 = eps2_1, eps4_1
                
                D_2 = Lambda_2_S*(eps2_2*(cell_Ip1J.W - cell_IJ.W) - eps4_2*(cell_Ip2J.W - 3*cell_Ip1J.W + 3*cell_IJ.W - cell_Im1J.W))
                D_3 = Lambda_3_S*(eps2_3*(cell_IJp1.W - cell_IJ.W) - eps4_3*(cell_IJp2.W - 3*cell_IJp1.W + 3*cell_IJ.W - cell_IJm1.W)) # Erreur D_3 pas bon car il prend en compte le W de la cellule J-1 (dummy cells) qui n<est pas une bonnne reference
                D_4 = Lambda_4_S*(eps2_4*(cell_Im1J.W - cell_IJ.W) - eps4_4*(cell_Im2J.W - 3*cell_Im1J.W + 3*cell_IJ.W - cell_Ip1J.W))
                D_1 = Lambda_1_S*(eps2_1*(cell_IJm1.W - cell_IJ.W) - eps4_1*(cell_IJm2.W - 3*cell_IJm1.W + 3*cell_IJ.W - cell_IJp1.W))
                
                cell_IJ.D_1 = D_1
                cell_IJ.D_2 = D_2
                cell_IJ.D_3 = D_3
                cell_IJ.D_4 = D_4
                
        for i in range(nx) : 
            #Swanson-Turkel
            # d2 = self.cells[4, i].W - 2*self.cells[3, i].W + self.cells[2, i].W 
            # d3 = self.cells[5, i].W - 4*self.cells[4, i].W + 5*self.cells[3, i].W - 2*self.cells[2, i].W
            
            # self.cells[3, i].D_3 = self.cells[4, i].D_1
            # self.cells[3, i].D_1 = d3 - self.cells[3, i].D_3
            
            # self.cells[2, i].D_3 = self.cells[3, i].D_1
            # self.cells[2, i].D_1 = d2 - self.cells[2, i].D_3
            
            self.cells[3, i].D_1 = self.cells[3,i].Lambda_1_S*(self.cells[3,i].eps2_1*(self.cells[2,i].W - self.cells[3,i].W) - self.cells[4,i].eps4_1*(-self.cells[2,i].W + 2*self.cells[3,i].W - self.cells[4,i].W))
            # print(self.cells[3, i].D_1)
            self.cells[2, i].D_1 = self.cells[2,i].Lambda_1_S*(self.cells[3,i].eps2_1*(self.cells[2,i].W - self.cells[3,i].W) - self.cells[4,i].eps4_1*(-self.cells[2,i].W + 2*self.cells[3,i].W - self.cells[4,i].W))
                                                                        
    
                
    def compute_R(self) :
        ny, nx = self.cells.shape
        for j in range(2, ny-2) :
            for i in range(nx) :
                FcDS_1 = self.cells[j, i].FcDS_1
                FcDS_2 = self.cells[j, i].FcDS_2
                FcDS_3 = self.cells[j, i].FcDS_3
                FcDS_4 = self.cells[j, i].FcDS_4
                
                D_1 = self.cells[j, i].D_1
                D_2 = self.cells[j, i].D_2
                D_3 = self.cells[j, i].D_3
                D_4 = self.cells[j, i].D_4
                
                # print(D_1 + D_2 + D_3 + D_4)
                
                self.cells[j, i].R = (FcDS_1-D_1  + FcDS_2-D_2 + FcDS_3-D_3 + FcDS_4-D_4)
                # print(f"R = {self.cells[j, i].R}")
                
    def run(self) :
        self.compute_dummy_cells()
        self.compute_Fc_DeltaS_Lamdac()
        self.compute_dissipation()
        self.compute_R()

class temporal_discretization :
    def __init__(self, x, y, rho, u, v, E, T, p, checkpoint_file_name=None) :
        self.x = x
        self.y = y
        self.rho = rho
        self.u = u
        self.v = v
        self.E = E
        self.T = T
        self.p = p
        self.checkpoint_file_name = checkpoint_file_name
        
        self.current_state = spatial_discretization(self.x, self.y, self.rho, self.u, self.v, self.E, self.T, self.p)
            
    
    def compute_dt(self, cell_IJ, sigma=0.25):
        rho, u, v, E, T, p = conservative_variable_from_W(cell_IJ.W)
        c = np.sqrt(1.4*287*T)
        
        n_I = 0.5*(cell_IJ.n2 - cell_IJ.n4)
        n_J = 0.5*(cell_IJ.n1 - cell_IJ.n3)
        Ds_I = 0.5*(cell_IJ.Ds2 + cell_IJ.Ds4)
        Ds_J = 0.5*(cell_IJ.Ds1 + cell_IJ.Ds3)
        
        lambda_I = (abs(u*n_I[0] + v*n_I[1]) + c)*Ds_I
        lambda_J = (abs(u*n_J[0] + v*n_J[1]) + c)*Ds_J
        
        dt = sigma*cell_IJ.OMEGA/(lambda_I + lambda_J)
        # print(dt)
        
        return dt
    
    def Runge_Kutta(self, it_max=100) :
        
        self.current_state.run()
        test_cell_1 = self.current_state.cells[4, 0]
        test_cell_2 = self.current_state.cells[5, 0]
        ny, nx = self.current_state.cells.shape
        print(ny, nx)
        
        it = 0
        
        if self.checkpoint_file_name is not None :
            q_cell, iteration, Residuals = load_checkpoint(self.checkpoint_file_name)
            
            for j in range(ny-4) :
                for i in range(nx) :
                    self.current_state.cells[j, i].W = q_cell[j, i]
            self.current_state.run()
            
            Residuals = Residuals
            iteration = iteration
            first_residual = Residuals[0, :]
            normalized_residuals = Residuals/first_residual
            print(f"{first_residual=}")
            
            # Initialize the plot
            plt.ion()  # Turn on interactive mode
            fig, axs = plt.subplots(2, 2)
            axs = axs.flatten()
            num_components = 4
            lines = []
            
            title = ["Mass", "Momentum x", "Momentum y", "Energy"]
            # Initialize each subplot
            for i in range(num_components):
                lines.append(axs[i].plot(iteration, normalized_residuals[:, i])[0])
                axs[i].set_title(f'{title[i]}')
                axs[i].set_xlabel('Iteration')
                axs[i].set_ylabel('Residuals L2 Norm')
                axs[i].set_yscale('log')  # Set y-axis to log scale
            plt.tight_layout()  # Adjust layout for better spacing
            plt.show()
            
            Residuals = list(Residuals)
            iteration = list(iteration)
            it = iteration[-1]
            
            
            
        else :
            Residuals = []
            iteration = []
            first_residual = None
        
            # Initialize the plot
            plt.ion()  # Turn on interactive mode
            fig, axs = plt.subplots(2, 2)
            axs = axs.flatten()
            num_components = 4
            lines = []
            
            title = ["Mass", "Momentum x", "Momentum y", "Energy"]
            # Initialize each subplot
            for i in range(num_components):
                lines.append(axs[i].plot(iteration, Residuals)[0])
                axs[i].set_title(f'{title[i]}')
                axs[i].set_xlabel('Iteration')
                axs[i].set_ylabel('Residuals L2 Norm')
                axs[i].set_yscale('log')  # Set y-axis to log scale
            plt.tight_layout()  # Adjust layout for better spacing
            plt.show()
            
        all_Res = np.ones((ny-4, nx, 4))
        all_dw = np.ones((ny-4, nx, 4))
        q = np.ones((ny-4, nx, 4))
        for j in range(2, ny-2) :
            for i in range(nx) :
                q[j-2, i, 0] = self.current_state.cells[j, i].W[0]
                q[j-2, i, 1] = self.current_state.cells[j, i].W[1]
                q[j-2, i, 2] = self.current_state.cells[j, i].W[2]
                q[j-2, i, 3] = self.current_state.cells[j, i].W[3]
        
        
        
        
    
        try:
            normalized_residuals = [1, 1, 1, 1]
            while np.max(normalized_residuals) > 1e-12 :
                # Stage 1
                for j in range(2, ny-2) :
                    for i in range(nx) :
                        dt = self.compute_dt(self.current_state.cells[j, i])
                        dW = 0.0533*dt*self.current_state.cells[j, i].R*(-1/self.current_state.cells[j, i].OMEGA)
                        # print(dW)
                        self.current_state.cells[j, i].W += dW
                        
                self.current_state.run()
                # Stage 2
                for j in range(2, ny-2) :
                    for i in range(nx) :
                        dt = self.compute_dt(self.current_state.cells[j, i])
                        dW = 0.1263*dt*self.current_state.cells[j, i].R*(-1/self.current_state.cells[j, i].OMEGA)
                        self.current_state.cells[j, i].W += dW
                self.current_state.run()
                # Stage 3
                for j in range(2, ny-2) :
                    for i in range(nx) :
                        dt = self.compute_dt(self.current_state.cells[j, i])
                        dW = 0.2375*dt*self.current_state.cells[j, i].R*(-1/self.current_state.cells[j, i].OMEGA)
                        self.current_state.cells[j, i].W += dW
                self.current_state.run()
                # Stage 4
                for j in range(2, ny-2) :
                    for i in range(nx) :
                        dt = self.compute_dt(self.current_state.cells[j, i])
                        dW = 0.4414*dt*self.current_state.cells[j, i].R*(-1/self.current_state.cells[j, i].OMEGA)
                        self.current_state.cells[j, i].W += dW
                self.current_state.run()
                # Stage 5
                for j in range(2, ny-2) :
                    for i in range(nx) :
                        dt = self.compute_dt(self.current_state.cells[j, i])
                        Res = self.current_state.cells[j, i].R
                        dW = dt*Res*(-1/self.current_state.cells[j, i].OMEGA)
                        self.current_state.cells[j, i].W += dW
                        all_Res[j-2, i, :] = Res
                        all_dw[j-2, i, :] = dW
                        q[j-2, i, 0] = self.current_state.cells[j, i].W[0]
                        q[j-2, i, 1] = self.current_state.cells[j, i].W[1]
                        q[j-2, i, 2] = self.current_state.cells[j, i].W[2]
                        q[j-2, i, 3] = self.current_state.cells[j, i].W[3]
                self.current_state.run()
                
                if it==900:
                    dw_test = np.copy(all_dw)
                    q_test = np.copy(q)
                elif it==901:
                    dw_test_1 = np.copy(all_dw)
                    q_test_1 = np.copy(q)
                elif it==700:
                    dw_test_2 = np.copy(all_dw)
                    q_test_2 = np.copy(q)
                    q_test_2 = np.copy(q)    
                    
                it += 1
                l2_norms = compute_L2_norm(np.abs(all_Res))
                # print(f"{all_Res=}")
                # print(f"{l2_norms=}")
                
                if first_residual is None:
                    first_residual = l2_norms  # Save the first residual
                normalized_residuals = l2_norms / first_residual
                # print(f"{normalized_residuals=}")
                Residuals.append(l2_norms)
                iteration.append(it)
                
                # Update each subplot with the new residual norms
                for i in range(num_components):
                    update_subplot(axs[i], normalized_residuals[i], it)
                
                print(f"Iteration {it}: L2 Norms = {normalized_residuals}")
                plt.draw()
                plt.pause(0.1)
                if it > it_max :
                    break
            
            q_cell_dummy = np.ones((ny-2, nx, 4))
            for j in range(1, ny-1) :
                for i in range(nx) :
                    q_cell_dummy[j-1, i, 0] = self.current_state.cells[j, i].W[0]
                    q_cell_dummy[j-1, i, 1] = self.current_state.cells[j, i].W[1]
                    q_cell_dummy[j-1, i, 2] = self.current_state.cells[j, i].W[2]
                    q_cell_dummy[j-1, i, 3] = self.current_state.cells[j, i].W[3]
            q_vertex = cell_dummy_to_vertex_centered_airfoil(q_cell_dummy)
            
            return q, q_vertex, Res#, q_test, q_test_1, q_test_2, dw_test, dw_test_1, dw_test_2
        
        except KeyboardInterrupt:
            print("Execution interrupted! Saving checkpoint...")
            
        finally:
            # Save the final state when the loop ends or is interrupted
            save_checkpoint(q, iteration, Residuals)
            
            q_cell_dummy = np.ones((ny-2, nx, 4))
            for j in range(1, ny-1) :
                for i in range(nx) :
                    q_cell_dummy[j-1, i, 0] = self.current_state.cells[j, i].W[0]
                    q_cell_dummy[j-1, i, 1] = self.current_state.cells[j, i].W[1]
                    q_cell_dummy[j-1, i, 2] = self.current_state.cells[j, i].W[2]
                    q_cell_dummy[j-1, i, 3] = self.current_state.cells[j, i].W[3]
            q_vertex = cell_dummy_to_vertex_centered_airfoil(q_cell_dummy)
            
            return q, q_vertex, Res
            
            
            
    def run(self) :
        q, q_vertex, Res = self.Runge_Kutta()
            
        return q, q_vertex, Res#, q_test, q_test_1, q_test_2, dw_test, dw_test_1, dw_test_2
            
            
      
        
if __name__ == "__main__":
        
    Mach = 0.8
    alpha = np.radians(2.0)
    p_inf = 1E5
    T_inf = 300
    
    a = np.sqrt(1.4*287*T_inf)
    Vitesse = Mach*a
    u = Vitesse*np.cos(alpha)
    v = Vitesse*np.sin(alpha)
    rho = p_inf/(T_inf*287)
    E = p_inf/((1.4-1)*rho) + 0.5*Vitesse**2
    
    x, y = read_PLOT3D_mesh("x.9")
    current_state = spatial_discretization(x, y, rho, u, v, E, T_inf, p_inf)
    current_state.run()
    domain_cell = current_state.domain_cells
    cell_test = current_state.cells[11, 0]
    
    ny, nx = x.shape
    for j in range(ny-1) :
        for i in range(nx-1) :
            print(current_state.cells[j, i].Lambda_1_I)
            # print(current_state.cells[j+2, i].Ds3)
            # print(current_state.cells[j+2, i].n4)
            # print(current_state.cells[j+2, i].R)
            # print(current_state.cells[j+2, i].FcDS_1 + current_state.cells[j+2, i].FcDS_2 + current_state.cells[j+2, i].FcDS_3 + current_state.cells[j+2, i].FcDS_4)
            # print("\n")
    
    # x, y = read_PLOT3D_mesh("x.6")
    # checkpoint_file_name="checkpoint_test.npz"
    # FEM = temporal_discretization(x, y, rho, u, v, E, T_inf, p_inf, checkpoint_file_name=None)
    # q, q_vertex, Res = FEM.run()
    
    
    # q_cell, iteration, Residuals = load_checkpoint()
    # q_vertex = cell_to_vertex_centered_airfoil(q_cell)
    
    # mu = 1.45*T_inf**(3/2)/(T_inf+110)*1e-6
    # reyn = rho*np.sqrt(u**2+v**2)*1/mu
    # write_plot3d_2d(x, y, q_vertex, Mach, alpha, reyn, 0., grid_filename="test.xy", solution_filename="test.q")
    
    
    
    
    # Mach = 0.5
    # alpha = np.radians(1.25)
    # p_inf = 1E5
    # T_inf = 288
    
    # a = np.sqrt(1.4*287*T_inf)
    # Vitesse = Mach*a
    # u = Vitesse*np.cos(alpha)
    # v = Vitesse*np.sin(alpha)
    # rho = p_inf/(T_inf*287)
    # E = p_inf/((1.4-1)*rho) + 0.5*Vitesse**2
        
    
    # x, y = read_PLOT3D_mesh("x.7")
    # checkpoint_file_name="checkpoint_test.npz"
    # FEM = temporal_discretization(x, y, rho, u, v, E, T_inf, p_inf, checkpoint_file_name=None)
    # q, q_vertex, Res = FEM.run()
    
    
    # # q_cell, iteration, Residuals = load_checkpoint()
    # # q_vertex = cell_to_vertex_centered_airfoil(q_cell)
    
    # mu = 1.45*T_inf**(3/2)/(T_inf+110)*1e-6
    # reyn = rho*np.sqrt(u**2+v**2)*1/mu
    # write_plot3d_2d(x, y, q_vertex, Mach, alpha, reyn, 0., grid_filename="test_M05_alpha125.xy", solution_filename="test_M05_alpha125.q")
    
    




















    
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