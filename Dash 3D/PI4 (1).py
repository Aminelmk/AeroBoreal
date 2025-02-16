import numpy as np
import matplotlib.pyplot as plt
import csv

# panel = [p1,p2,p3,p4] où ces points sont définis en sens horaire  (pi = [x,y,z])
# p = [x,y,z], point de collocation
# n = [x,y,z], vecteur normal au point de collocation p


def Vortxl(p, p1, p2, gamma=1):
    
    r0 = [p2[0]-p1[0],p2[1]-p1[1],p2[2]-p1[2]]
    r1 = [p[0]-p1[0],p[1]-p1[1],p[2]-p1[2]]
    r2 = [p[0]-p2[0],p[1]-p2[1],p[2]-p2[2]]
    
    r1_2 = (r1[0]**2 + r1[1]**2 + r1[2]**2)**0.5
    r2_2 = (r2[0]**2 + r2[1]**2 + r2[2]**2)**0.5
    
    r1xr2 = np.cross(r1,r2)
    r1xr2_2 = r1xr2[0]**2 + r1xr2[1]**2 + r1xr2[2]**2

    denom = 4*np.pi*r1xr2_2
    terme1 = np.dot(r0,r1)/r1_2
    terme2 = np.dot(r0,r2)/r2_2
    K = gamma/denom*(terme1 - terme2)
    
    return [K*r1xr2[0],K*r1xr2[1],K*r1xr2[2]]


def Voring(p, panel, gamma=1):
    
    u1 = Vortxl(p, panel[0], panel[1], gamma)
    u2 = Vortxl(p, panel[1], panel[2], gamma)
    u3 = Vortxl(p, panel[2], panel[3], gamma)
    u4 = Vortxl(p, panel[3], panel[0], gamma)
    
    u = u1[0] + u2[0] + u3[0] + u4[0]
    v = u1[1] + u2[1] + u3[1] + u4[1]
    w = u1[2] + u2[2] + u3[2] + u4[2]
    
    ulon = u1[0] + u3[0] 
    vlon = u1[1] + u3[1] 
    wlon = u1[2] + u3[2] 
    
    return [u,v,w], [ulon,vlon,wlon]


def norm(panel):
    
    A = panel[2] - panel[0]
    B = panel[1] - panel[3]
    n = np.cross(A,B)
    n /= np.linalg.norm(n)
    
    return n


def coefA(p, panel, n, sym=True):
    
    if sym == True:
        panelSym = np.zeros([4,3])
        panelSym[:,:] = panel[:,:]
        panelSym[:,1] = panelSym[:,1]*-1
        a_ij = np.dot(np.add(Voring(p,panel)[0],Voring(p,panelSym,-1)[0]),n)
    else:
        a_ij = np.dot(Voring(p,panel)[0],n)

    return a_ij


def coefB(p, panel, n):
    
    b_ij = np.dot(Voring(p,panel)[1],n)

    return b_ij


def funcRHS(U_inf, n):
    
    RHS = -np.dot(U_inf,n)
    
    return RHS


def colloc(panel):
    
    return 0.25*(panel[0] + panel[1] + panel[2] + panel[3])


def aire(panel):
    
    a = np.cross(panel[1]-panel[0],panel[2]-panel[0])
    b = np.cross(panel[2]-panel[0],panel[3]-panel[0])
    
    return 0.5*(np.linalg.norm(a) + np.linalg.norm(b))



def maillage(ny, nx, span, cord, alpha, glisse=0, sweep=0, lam=1, diedre=0):
    
    x1 = np.zeros([nx+1,ny+1])
    y1 = np.zeros([nx+1,ny+1])
    z1 = np.zeros([nx+1,ny+1])
    xlinspace = np.linspace(0,cord,nx+1)
    # ylinspace = np.linspace(0,span,ny+1)
    theta = np.linspace(np.pi/2,np.pi,ny+1)
    ylinspace = -1*np.cos(theta)*span
    zlinspace = np.linspace(0,span*np.tan(np.deg2rad(diedre)),ny+1)
    for j in range(ny+1):
        for i in range(nx+1):
            y1[i,j] = ylinspace[j]
    for i in range(nx+1):
        for j in range(ny+1):
            x1[i,j] = xlinspace[i] + y1[i,j]*np.tan(np.deg2rad(sweep)) + y1[i,j]/span*cord*(1-lam)/2*(1 - 2*i/nx)
    for j in range(ny+1):
        for i in range(nx+1):
            z1[i,j] = zlinspace[j]

    x2 = np.zeros([nx+1,ny+1])
    y2 = np.zeros([nx+1,ny+1])
    z2 = np.zeros([nx+1,ny+1])
    for j in range(ny+1):
        for i in range(nx+1):
            y2[i,j] = y1[i,j]
    for i in range(nx):
        for j in range(ny+1):
            x2[i,j] = x1[i,j] + 0.25*(x1[i+1,j] - x1[i,j])
    for j in range(ny+1):
        x2[nx,j] = x1[nx,j] + 0.25*(x1[nx,j] - x1[nx-1,j])
    for i in range(nx):
        for j in range(ny+1):
            z2[i,j] = z1[i,j] + 0.25*(z1[i+1,j] - z1[i,j])
    for j in range(ny+1):
        z2[nx,j] = z1[nx,j] + 0.25*(z1[nx,j] - z1[nx-1,j])

    x2W = np.zeros([1,ny+1])
    y2W = np.zeros([1,ny+1])
    z2W = np.zeros([1,ny+1])
    for j in range(ny+1):
        x2W[0,j] = x2[nx,j] + 100*cord*np.cos(np.deg2rad(alpha))*np.cos(np.deg2rad(glisse))
        y2W[0,j] = ylinspace[j] + 100*cord*np.sin(np.deg2rad(glisse))*np.cos(np.deg2rad(alpha))
        z2W[0,j] = z2[nx,j] + 100*cord*np.sin(np.deg2rad(alpha))


    panels1 = np.zeros([nx,ny,4,3])
    panels2 = np.zeros([nx,ny,4,3])
    panels2W = np.zeros([1,ny,4,3])
    for i in range(nx):
        for j in range(ny):
            p1 = [x1[i,j+1],y1[i,j+1],z1[i,j+1]]
            p2 = [x1[i+1,j+1],y1[i+1,j+1],z1[i+1,j+1]]
            p3 = [x1[i+1,j],y1[i+1,j],z1[i+1,j]]
            p4 = [x1[i,j],y1[i,j],z1[i,j]]
            panels1[i,j] = [p1,p2,p3,p4]
    for i in range(nx):
        for j in range(ny):
            p1 = [x2[i,j+1],y2[i,j+1],z2[i,j+1]]
            p2 = [x2[i+1,j+1],y2[i+1,j+1],z2[i+1,j+1]]
            p3 = [x2[i+1,j],y2[i+1,j],z2[i+1,j]]
            p4 = [x2[i,j],y2[i,j],z2[i,j]]
            panels2[i,j] = [p1,p2,p3,p4]
    for j in range(ny):
        p1 = [x2[nx,j+1],y2[nx,j+1],z2[nx,j+1]]
        p2 = [x2W[0,j+1],y2W[0,j+1],z2W[0,j+1]]
        p3 = [x2W[0,j],y2W[0,j],z2W[0,j]]
        p4 = [x2[nx,j],y2[nx,j],z2[nx,j]]
        panels2W[0,j] = [p1,p2,p3,p4]


    panels1out = np.zeros([nx*ny,4,3])
    panels2out = np.zeros([nx*ny,4,3])
    panels2Wout = np.zeros([ny,4,3])
    for i in range(nx):
        for j in range(ny):
            panels1out[i*ny+j] = panels1[i,j]
            panels2out[i*ny+j] = panels2[i,j]
    for j in range(ny):
        panels2Wout[j] = panels2W[0,j]
        
    for i in range(len(panels1out)):
        plt.plot(panels1out[i][:,1],panels1out[i][:,0])
        print(panels1out[i][:,1],panels1out[i][:,0])
    plt.show()
    
    # Collect panel coordinates for output
    panel_coords = []
    for panel in panels1out:
        for point in panel:
            panel_coords.append(point)
    
    return panels1out, panels2out, panels2Wout, np.array(panel_coords)


def delta_y(panel):

    return np.linalg.norm(panel[3][1]-panel[0][1])
    
def delta_y_vec(panel):
    
    return panel[3]-panel[0]



def lecture_Euler(Cl):
    
    nbr_fichiers = 3
    if nbr_fichiers == 1:
        position_span = 1
    else:
        position_span = np.linspace(0,1,nbr_fichiers)
    
    for i in range(nbr_fichiers):
        pass
    
 
    return 


# def ecriture(p_col,p_ij):
    
#     data = np.zeros([len(p_ij),4])
#     for i in range(len(p_ij)):
#         data[i,:] = [p_col[0],p_col[1],p_col[2],p_ij]
#     nom_fichier = 'Pressions.csv'
#     with open(nom_fichier, mode='w', newline='') as fichier:
#         writer = csv.writer(fichier)
#         writer.writerow(["x","y","z","Pression"])
#         for point in data:
#             writer.writerow(point)
    
#     return

def output_maillage_to_csv(panel_coords, filename="message.txt"):
    # Open the text file for writing
    with open(filename, mode='w') as file:
        # Write the panel coordinates
        for point in panel_coords:
            line = ','.join(map(str, point)) + ',\n'
            file.write(line)
    
    print(f"Maillage panel coordinates have been written to {filename}.")








def main(AR,alpha,ny,nx,sweep=0,lam=1,diedre=0,glisse=0):
    
    if type(alpha) == list:
        if len(alpha) != ny:
            print("alpha doit être de même taille que le nombre de panneaux le long de l'envergure")
            pass
    else:
        alphaMaillage = alpha
        alpha = np.array([float(alpha)])
        
    
    
    rho = 1
    Q_inf = 1
    # alpha = 5

    n_LE = ny
    corde = 1
    span = AR*(corde+lam)/4
    # glisse = 0
    # sweep = 0
    # lam = 1
    _, panel, panelW, panel_coords = maillage(n_LE, nx, span, corde, alphaMaillage, glisse, sweep, lam, diedre)
    n = len(panel)
    
    U_inf = np.zeros([n_LE,3])
    if len(alpha) == 1:
        for i in range(n_LE):
            U_inf[i] = [Q_inf*np.cos(np.deg2rad(alpha[0]))*np.cos(np.deg2rad(glisse)), Q_inf*np.sin(np.deg2rad(glisse))*np.cos(np.deg2rad(alpha[0])), Q_inf*np.sin(np.deg2rad(alpha[0]))]
    else:
        for i in range(n_LE):
            U_inf[i] = [Q_inf*np.cos(np.deg2rad(alpha[i]))*np.cos(np.deg2rad(glisse)), Q_inf*np.sin(np.deg2rad(glisse))*np.cos(np.deg2rad(alpha[i])), Q_inf*np.sin(np.deg2rad(alpha[i]))]
    a_ij = np.zeros([n,n])
    b_ij = np.zeros([n,n])
    RHS = np.zeros(n)
    p_colloc = np.zeros([n,3])
    n_panel = np.zeros([n,3])
    for i in range(n):
        p_colloc[i] = colloc(panel[i])
        n_panel[i] = norm(panel[i])
    for i in range(n):
        for j in range(n-n_LE):
            a_ij[i,j] = coefA(p_colloc[i],panel[j],n_panel[j])
            b_ij[i,j] = coefB(p_colloc[i],panel[j],n_panel[j])
        indW = 0
        for j in range(n-n_LE,n):
            a_ij[i,j] = coefA(p_colloc[i],panel[j],n_panel[j]) + coefA(p_colloc[i],panelW[indW],n_panel[j])
            b_ij[i,j] = coefB(p_colloc[i],panel[j],n_panel[j]) + coefB(p_colloc[i],panelW[indW],n_panel[j])
            indW += 1  
    if len(alpha) == 1:
        for i in range(n):
            RHS[i] = funcRHS(U_inf[0], n_panel[i])
    else:
        for i in range(n):
            RHS[i] = funcRHS(U_inf[i%n_LE], n_panel[i])
        
    gamma = np.linalg.solve(a_ij,RHS)
    w_ind = np.matmul(b_ij,gamma)

    aire_ij = np.zeros(n)
    dy_ij = np.zeros(n)
    dy_ij_vec = np.zeros([n,3])
    L_ij = np.zeros(n)
    p_ij = np.zeros(n)
    D_ij = np.zeros(n)
    for i in range(n_LE):
        aire_ij[i] = aire(panel[i])
        dy_ij[i] = delta_y(panel[i])
        dy_ij_vec[i] = delta_y_vec(panel[i])
        # L_ij[i] = rho*gamma[i]*np.linalg.norm(U_inf[i,:])*dy_ij[i]
        # print(dy_ij_vec[i,:],np.linalg.norm(np.cross(U_inf[i,:],dy_ij_vec[i,:])),np.linalg.norm(U_inf[i,:])*dy_ij[i])
        L_ij[i] = rho*gamma[i]*np.linalg.norm(np.cross(U_inf[i,:],dy_ij_vec[i,:]))
        p_ij[i] = L_ij[i]/aire_ij[i]
        D_ij[i] = -rho*gamma[i]*w_ind[i]*dy_ij[i]
    for i in range(n_LE,n):
        aire_ij[i] = aire(panel[i])
        dy_ij[i] = delta_y(panel[i])
        dy_ij_vec[i] = delta_y_vec(panel[i])
        # L_ij[i] = rho*(gamma[i] - gamma[i-n_LE])*np.linalg.norm(U_inf[i%n_LE,:])*dy_ij[i]
        L_ij[i] = rho*(gamma[i] - gamma[i-n_LE])*np.linalg.norm(np.cross(U_inf[i%n_LE,:],dy_ij_vec[i,:]))
        p_ij[i] = L_ij[i]/aire_ij[i]
        D_ij[i] = -rho*(gamma[i] - gamma[i-n_LE])*w_ind[i]*dy_ij[i]
    
    A = np.sum(aire_ij)   # 2 ailes 
    L = np.sum(L_ij)
    D = np.sum(D_ij)
    CL = L/(0.5*rho*Q_inf**2*A)   # la portance est multipliée par 2 puisque la portance totale est celle des 2 ailes
    CD = D/(0.5*rho*Q_inf**2*A)/2
    
    Cl = np.zeros(ny)
    aire_span = 0
    for i in range(ny):
        for j in range(nx):
            Cl[i] += L_ij[j*ny+i]
            aire_span += aire_ij[j*ny+i]
        Cl[i] /= (0.5*rho*np.linalg.norm(U_inf[i,:])**2*aire_span)
        aire_span = 0
    
    X = np.linspace(0,1,nx)
    Y = np.linspace(-span,span,n_LE)
    pression3d = np.zeros([n_LE,nx])
    for i in range(nx):
        for j in range(n_LE):
            pression3d[j][i] = p_ij[i*n_LE+j]
            
    
    output_maillage_to_csv(panel_coords)
        
    return CL, Cl, CD


AR = 40
alpha = 10
ny = 20
nx = 4
sweep = 10
lam = 1
diedre = 0
glisse = 5
CL,Cl,CD = main(AR,alpha,ny,nx,sweep,lam,diedre,glisse)
print(CL,CD)












