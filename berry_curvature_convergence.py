# -*- coding: utf-8 -*-
"""
Created on Sat Sep 27 13:11:49 2025

@author: Gargi Joshi
"""

import numpy as np
import matplotlib.pyplot as plt
import tidy3d as td
from shapely.geometry import Polygon, Point, MultiPolygon
from shapely.affinity import translate
import matplotlib.colors
import time
from matplotlib.patches import Rectangle
from matplotlib.patches import Polygon as poly
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.path import Path

def make_regular_polygon(n, x_centre=0, y_centre=0, radial=False, radial_distance=0, side_length=0, rotation_angle=0):
    '''
    
    Parameters
    ----------
    n : integer
        Number of sides of polygon.
    x_centre : float
        x-coordinate of centre.
    y_centre : float
        y-coordinate of centre.
    radial_distance : float
        radial distance
    side_length: float
        length of side
    rotation_angle : float
        angle by which the polygon is rotated, given that 
        the first point starts from (r, 0)

    Returns 
    -------
    (x, y) ->vertices of the polygon

    '''
    vertices=np.zeros((n, 2))
    angle=np.linspace(0, 2*np.pi, n, endpoint=False)
    angle+=(np.pi/2 if n%2 else np.pi/n)
    angle+=rotation_angle
    if (radial):
        r=radial_distance
    else:
        r=side_length/(2*np.sin(np.pi/n))
    
    for i in range(n):
        vertices[i][0]=x_centre+r*np.cos(angle[i])
        vertices[i][1]=y_centre+r*np.sin(angle[i])
    #vertices = np.column_stack((x, y))
    #print(vertices)
    return vertices

def initialize_lattice_parameters(a):
    a=a
    c0=td.C_0
        
    #reciprocal lattice vectors
    B1=2*np.pi/a *np.array([1, 1/np.sqrt(3)])
    B2=2*np.pi/a *np.array([1, -1/np.sqrt(3)])
    
    return a, c0, B1, B2

def initialize_hole_parameters(a, ratio_1, ratio_2):
    l1=a*ratio_1
    l2=a*ratio_2
    
    return l1, l2

def one_unit_cell (n, a, a1, a2, x_centre_1=0, y_centre_1=0, x_centre_2=0, y_centre_2=0, rotation_angle_1=0, rotational_angle_2=np.pi, radial=False, symmetry_seperation=0):
    '''

    Parameters
    ----------
    n : number of sides of inside polygons
    a : lattice constant/radius of circle in which polygon lies
    a1 : radius/length of one polygon
    a2 : radius/length of 2nd polygon
    x_centre_1 : centre of 1st polygon x coord
    y_centre_1 : centre of 1st polygon y coord
    x_centre_2 : centre of 2nd polygon x coord
    y_centre_2 : Tcentre of 2nd polygon y coord
    rotation_angle : angle rotated through 
    radial : radial polygon or side length
        The default is False.
    radial_distance : 
        '''
    if (radial):
        r1=a1
        r2=a1
        l1=0 
        l2=0
    else:
        r1=0
        r2=0 
        l1=a1 
        l2=a2
        
    poly_1=make_regular_polygon(n=n, x_centre=x_centre_1, y_centre=y_centre_1, radial=radial, radial_distance=r1, side_length=l1, rotation_angle=rotation_angle_1)
    polygon_1=Polygon(poly_1)
    poly_2=make_regular_polygon(n=n, x_centre=x_centre_2, y_centre=y_centre_2, radial=radial, radial_distance=r2, side_length=l2, rotation_angle=rotational_angle_2)
    polygon_2=Polygon(poly_2)
    
    unit = translate(polygon_1, 0, a/(2 * np.sqrt(3))).union(translate(polygon_2, 0, -a/(2 * np.sqrt(3)))) #couldn't understand the exact reason behind these particular coords
    unit = unit.union(translate(unit,  a/2, a*np.sqrt(3)/2))
    unit = unit.union(translate(unit, -a/2, a*np.sqrt(3)/2))
    unit = translate(unit, 0, -np.sqrt(3)*a/2)
    return unit

def dielectric_function(ed, ea, unit, a, x_start, x_end, y_start, y_end, N_sp, tolerance=0,  rhombus=True):
    
    n1 = np.linspace(x_start, x_end, 2*N_sp, endpoint=False)
    n2 = np.linspace(y_start, y_end, 2*N_sp, endpoint=False)
    
    N1, N2 = np.meshgrid(n1, n2)
   
    X = a * N1
    Y = a * np.sqrt(3) * N2
    
    if (rhombus==True):
        dx = a / (2 * N_sp)
        for i in range(2 * N_sp):
            X[i, :] += dx / 2 * (i)
    
    
    xi = X.reshape((X.size,1), order = 'F')
    yi = Y.reshape((Y.size,1), order = 'F')
   
    
    unit_expanded = unit.buffer(tolerance)    
    #unit_contracted = unit.buffer(-tolerance)
    
    def is_inside_unit_polygon(x, y):
        point = Point(x, y)  
        return unit_expanded.contains(point)
    
    is_inside_vec = np.vectorize(is_inside_unit_polygon)
    eps = np.where(is_inside_vec(xi, yi), ea, ed)
    inv_eps=1/eps
    '''
    plt.close()
    plt.figure(figsize=(8, 6))
    if (rhombus==True):
        plt.axes().set_aspect(1)
    else:
        plt.axes().set_aspect(0.5)
    plt.scatter(xi, yi, c=inv_eps, s=1, cmap='jet')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.colorbar(label=r'$\epsilon(x, y)$')
    plt.title('Spatial Dielectric Distribution')
    plt.show()
'''
    return inv_eps, xi, yi

def specify_dielectric_function_rectangle(a, unit, N_sp):

    ed = 9  # Relative permeability of shaded region (dielectric)
    ea = 1  # Relative permeability of white space (air)
    n1 = np.linspace(-0.5, 0.5, 2*N_sp, endpoint=False)
    n2 = np.linspace(-0.5, 0.5, 2*N_sp, endpoint=False)
  
    N1, N2 = np.meshgrid(n1, n2)

    # Generating spatial mesh
    X = a * N1
    Y = a * np.sqrt(3) * N2
    xi = X.reshape((X.size,1), order = 'F')
    yi = Y.reshape((Y.size,1), order = 'F')

    # Define a small tolerance value for boundary checks
    tolerance = 1e-8 * a  # This can be adjusted based on the scale of the problem

    # Buffer the polygon slightly to create a tolerance zone
    unit_expanded = unit.buffer(tolerance)    # Slightly expand the polygon
    unit_contracted = unit.buffer(-tolerance) # Slightly contract the polygon

    def is_inside_unit_polygon(x, y):
        point = Point(x, y)
        # Check if the point is inside the expanded polygon but not outside the contracted one
        return unit_expanded.contains(point)

    # Vectorizing the function for array operations
    is_inside_unit_polygon_vec = np.vectorize(is_inside_unit_polygon)

    # Recalculating the spatial dielectric distribution "exy" and its inverse "inv_exy"
    exy = np.where(is_inside_unit_polygon_vec(xi, yi), ea, ed)
    inv_exy = 1 / exy
    print(inv_exy[len(inv_exy)//2])
    exy_reshaped_1 = inv_exy.reshape(N1.shape, order = 'F')
    '''
    plt.close()
    plt.figure(figsize=(8, 6))
    plt.axes().set_aspect(0.5)
    plt.scatter(xi, yi, c=inv_exy, s=1, cmap='jet')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.colorbar(label=r'$\epsilon(x, y)$')
    plt.title('Spatial Dielectric Distribution')
    plt.show()
    '''
    return xi, yi, inv_exy, exy

def fourier_coefficients_reshaped(a, ng, B1, B2, xi, yi, inv_eps):
    m_max, n_max = 2 * ng, 2 * ng
    m = np.arange(-m_max, m_max + 1)
    n = np.arange(-n_max, n_max + 1)
    M, N = np.meshgrid(m, n)

    G_len=M.size
    
    M_lin = M.reshape((G_len, 1), order='C')
    N_lin = N.reshape((G_len, 1), order='C')
    
    ni=len(xi)
    
    chi = np.zeros((G_len, 1), dtype=complex)
    for i in range(G_len):
       phi = ((B1[0]*M_lin[i] + B2[0]*N_lin[i]) * xi) + ((B1[1]*M_lin[i] + B2[1]*N_lin[i]) * yi)
       chi[i] = np.sum(inv_eps * np.exp(-1j * phi)) / ni

    chi_matrix = chi.reshape(M.shape, order='F')
    '''
    # Plot chi(G)
    plt.figure()
    plt.imshow(np.abs(chi_matrix), extent=[-m_max-0.5, m_max+0.5, -n_max-0.5, n_max+0.5],
               norm=matplotlib.colors.LogNorm())
    plt.colorbar(label=r'$\chi(G)$')
    plt.title('Chi(G) Matrix')
    plt.set_cmap('jet')
    plt.show()
'''
    
    #chi(G-G')
    mp = np.arange(-ng, ng + 1)
    np_arr = np.arange(-ng, ng + 1)
    Np, Mp = np.meshgrid(np_arr, mp)

    G_lenp = Mp.size
    Mp_lin = Mp.reshape(G_lenp, order='F')
    Np_lin = Np.reshape(G_lenp, order='F')

    
    chi_p = []
    
    for i in range(G_lenp):
        crop = (np.abs(M_lin + Mp_lin[i]) <= ng) & (np.abs(N_lin + Np_lin[i]) <= ng)
        ### didnt understand why
        chi_p.append(chi[crop])

    chi_p = np.column_stack(chi_p)

    # Plot Chi(G-G')
    '''
    plt.figure()
    plt.imshow(np.abs(chi_p), norm=matplotlib.colors.LogNorm())
    plt.xlabel("G'")
    plt.ylabel("G")
    plt.colorbar(label=r'$\chi(G-G)$')
    plt.title("Chi(G-G') Matrix")
    plt.show()
'''
    return chi_p, chi, M_lin, N_lin, Mp_lin, Np_lin

def verify_dielectric(chi, M_lin, N_lin, xi, yi, B1, B2):
    
    G_len=len(M_lin)
    ni=len(xi)

    ft = np.zeros(ni, dtype=complex)

    for j in range(ni):
        x = xi[j]
        y = yi[j]
        phi = (B1[0]*M_lin + B2[0]*N_lin) * x + (B1[1]*M_lin + B2[1]*N_lin) * y
        ft[j] = np.sum(chi.flatten(order='C') * np.exp(1j * phi.flatten()))
        #print(f"{j/ni*100}% done")
    inv_eps_r = 1 / ft


    plt.figure(figsize=(8, 6))
    plt.axes().set_aspect(0.5)
    plt.scatter(xi, yi, c=np.real(ft), s=1, cmap='jet')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.colorbar(label=r'$\epsilon^{-1}(x, y)$')
    plt.title('Inverse Fourier of Dielectric Distribution')
    plt.show()

#1'       
def initialize_BZ_parameters(a, numG, num_BZ, band_index):

    kx = np.linspace(-2.4 * np.pi/a, 2.4 * np.pi/a, 2 * num_BZ) #wave vectors
    ky = np.linspace(-2.4 * np.pi/a, 2.4 * np.pi/a, 2 * num_BZ)
    KX, KY = np.meshgrid(kx, ky) 
    KX_lin = KX.reshape((KX.size, 1), order='F')
    KY_lin = KY.reshape((KY.size, 1), order='F')
    delta_kx = kx[1] - kx[0] 
    delta_ky = ky[1] - ky[0]
    band_index = band_index
    del_S = delta_kx * delta_ky
    return kx, ky, KX, KY, KX_lin, KY_lin, delta_kx, delta_ky, band_index, del_S


#2

def compute_berry_curvature(n, KX, KY, del_S, Hz_n_k, N_BZ, Mp_lin, Np_lin, B1, B2, khi_G_Gp, a, numG):
    KX_lin = KX.flatten()
    KY_lin = KY.flatten()
    H = [] #will store eigenvectors for each k-point
    dispe = np.zeros((numG, len(KX_lin))) #dispersion initialization (stores eig val)
    for i in range(len(KX_lin)): #to compute bloch eigenstates
    #h=eigen vector for n bands
        h, _ = Hz_n_k(n, [KX_lin[i], KY_lin[i]], i, Mp_lin, Np_lin, B1, B2, khi_G_Gp, a, dispe)
        H.append(h)
        #print(f"part1: {i / len(KX_lin) * 100:.2f}% completed", end='\r')

    H = np.array(H) 
    sz = KX.shape  #shape of k-grid
    # U=to store products of overlaps around each plaquette in k-space
    U = np.zeros((sz[0] - 1, sz[1] - 1), dtype=complex) #periodic bloch func
    
    for i in range(sz[0] - 1): #loop over each plaquette (grid in k-space)
        for j in range(sz[1] - 1):
            #indexing
            #Each small square plaquette in the k-grid is bordered by 4 points.
            id1 = np.ravel_multi_index((i, j), sz)
            id2 = np.ravel_multi_index((i, j + 1), sz)
            id3 = np.ravel_multi_index((i + 1, j + 1), sz)
            id4 = np.ravel_multi_index((i + 1, j), sz)
            
            #It computes the overlaps of neighboring Bloch states around the 
            #edges of this plaquette.
            #normalize
            #This step ensures gauge invariance (phase normalization).
            #These represent parallel transport of the Bloch function 
            #around the edges of the plaquette
            
            U1 = np.sum(np.conj(H[id1, :]) * H[id2, :]) #overlap 1 and 2
            U1 /= np.abs(U1)
            
            U2 = np.sum(np.conj(H[id2, :]) * H[id3, :])
            U2 /= np.abs(U2)
            
            U3 = np.sum(np.conj(H[id3, :]) * H[id4, :])
            U3 /= np.abs(U3)
            
            U4 = np.sum(np.conj(H[id4, :]) * H[id1, :])
            U4 /= np.abs(U4)
            
            U[i, j] = U1 * U2 * U3 * U4 #accumulated berry phase around a single plaquette
    
        #print(f"part2: {i / (sz[0] - 1) * 100:.2f}% completed", end='\r')
    
    F = np.imag(np.log(U)) / del_S 
    #Complex log of the Wilson loop product, gives Berry phase.
    #np.imag=>Extract the Berry curvature contribution (phase) per plaquette.
    #/delS to get curvature density
    return F, H, dispe

'''
def compute_berry_curvature_k(n, KX, KY, k1, k2, del_S, Hz_n_k, N_BZ, Mp_lin, Np_lin, B1, B2, khi_G_Gp, a, numG):
    
    ny, nx = KX.shape

    div_x=KX[0, :]
    div_y=KY[:, 0]
    i = np.argmin(np.abs(div_x - k1)) #column index
    j = np.argmin(np.abs(div_y - k2)) #row index

    k_corners=[(KX[j, i], KY[j, i]), (KX[j+1, i], KY[j+1, i]), (KX[j+1, i+1], KY[j+1, i+1]), (KX[j, i+1], KY[j, i+1])]

    dispe = np.zeros((numG))
    H = []

    for ka, kb in k_corners:
        h, _ = Hz_n_k(n, (ka, kb), 1, Mp_lin, Np_lin, B1, B2, khi_G_Gp, a, dispe)
        H.append(h)

    H = np.array(H) 
    sz = KX.shape  #shape of k-grid

    def normalized_overlap(a, b):
        ov = np.vdot(a, b)
        mag = np.abs(ov)
        if mag < 1e-15:   # avoid division by zero
            return 1.0 + 0.0j
        return ov / mag

    U1 = normalized_overlap(H[0], H[1])
    U2 = normalized_overlap(H[1], H[2])
    U3 = normalized_overlap(H[2], H[3])
    U4 = normalized_overlap(H[3], H[0])

    Uval = U1 * U2 * U3 * U4
    Fval = np.imag(np.log(Uval)) / del_S 
    #Complex log of the Wilson loop product, gives Berry phase.
    #np.imag=>Extract the Berry curvature contribution (phase) per plaquette.
    #/delS to get curvature density
    U = np.zeros((ny - 1, nx - 1), dtype=complex)
    F = np.zeros((ny - 1, nx - 1), dtype=float)
    U[j, i] = Uval
    F[j, i] = Fval

    return F, H, dispe, i, j
'''

def compute_berry_curvature_k(n, KX, KY, k1, k2, del_S, Hz_n_k,
                              N_BZ, Mp_lin, Np_lin, B1, B2, khi_G_Gp, a, numG):
    """
    Computes Berry curvature for a given k-point using Wilson loop method,
    ensuring double precision (float64 / complex128) throughout.
    """

    # ---- Enforce double precision on all inputs ----
    KX = np.asarray(KX, dtype=np.float64)
    KY = np.asarray(KY, dtype=np.float64)
    B1 = np.asarray(B1, dtype=np.float64)
    B2 = np.asarray(B2, dtype=np.float64)
    khi_G_Gp = np.asarray(khi_G_Gp, dtype=np.complex128)
    a = np.float64(a)
    del_S = np.float64(del_S)

    ny, nx = KX.shape

    # ---- Identify plaquette corners ----
    div_x = KX[0, :].astype(np.float64)
    div_y = KY[:, 0].astype(np.float64)

    i = np.argmin(np.abs(div_x - k1))
    j = np.argmin(np.abs(div_y - k2))

    k_corners = [
        (KX[j, i],     KY[j, i]),
        (KX[j+1, i],   KY[j+1, i]),
        (KX[j+1, i+1], KY[j+1, i+1]),
        (KX[j, i+1],   KY[j, i+1])
    ]

    # ---- Allocate arrays with explicit dtypes ----
    dispe = np.zeros(numG, dtype=np.float64)
    H = []

    # ---- Collect eigenstates at each plaquette corner ----
    for ka, kb in k_corners:
        h, _ = Hz_n_k(n, (ka, kb), 1, Mp_lin, Np_lin, B1, B2, khi_G_Gp, a, dispe)
        h = np.asarray(h, dtype=np.complex128)
        H.append(h)

    H = np.array(H, dtype=np.complex128)

    # ---- Helper: normalized overlap ----
    def normalized_overlap(a, b):
        a = np.asarray(a, dtype=np.complex128)
        b = np.asarray(b, dtype=np.complex128)
        ov = np.vdot(a, b).astype(np.complex128)
        mag = np.abs(ov)
        if mag < 1e-15:
            return np.complex128(1.0 + 0.0j)
        return ov / mag

    # ---- Compute link variables (Wilson loop) ----
    U1 = normalized_overlap(H[0], H[1])
    U2 = normalized_overlap(H[1], H[2])
    U3 = normalized_overlap(H[2], H[3])
    U4 = normalized_overlap(H[3], H[0])

    Uval = np.complex128(U1 * U2 * U3 * U4)

    # ---- Compute Berry curvature per plaquette ----
    Fval = np.imag(np.log(Uval)).astype(np.float64) / del_S

    # ---- Allocate full curvature arrays ----
    U = np.zeros((ny - 1, nx - 1), dtype=np.complex128)
    F = np.zeros((ny - 1, nx - 1), dtype=np.float64)

    # ---- Store computed values ----
    U[j, i] = Uval
    F[j, i] = Fval

    return F, H, dispe, i, j


def berry_curvature_local(KX, KY, F, kx, ky):
    div_x=KX[0, :]
    div_y=KY[:, 0]
    id_x = np.argmin(np.abs(div_x - kx))
    id_y = np.argmin(np.abs(div_y - ky))
    
    return F[id_x, id_y]
    

#F: Berry curvature over the grid.
#H: Eigenvectors at each k-point.
#dispe: Energy dispersion.

def compute_eigenstates_and_eigenfrequencies(n, k, i_index, Mp_lin, Np_lin, B1, B2, khi_G_Gp, a, dispe):
    MM, MMP = np.meshgrid(Mp_lin, Mp_lin) #MM and MMP are transpose of each other
    NN, NNP = np.meshgrid(Np_lin, Np_lin)
    
    Gx = B1[0] * MM + B2[0] * NN #for rows
    Gy = B1[1] * MM + B2[1] * NN
    
    Gpx = B1[0] * MMP + B2[0] * NNP #for columns
    Gpy = B1[1] * MMP + B2[1] * NNP
    
    G_k_Gp_k = ((Gx + k[0]) * (Gpx + k[0])) + ((Gy + k[1]) * (Gpy + k[1])) 
    #Builds the Hamiltonian matrix theta in G-space.
    theta = khi_G_Gp * G_k_Gp_k
    
    w, V = np.linalg.eig(theta)
    w = np.real(w)
    IX = np.argsort(w)
    w = w[IX]
    
    dispe[:] = np.sqrt(w) * a / (2 * np.pi)
    
    w_n = w[n - 1]
    w_n = np.real(np.sqrt(w_n) * (a / (2 * np.pi)))
    Hznk = V[:, IX[n - 1]]
    
    i_index = i_index + 1
    
    return Hznk, w_n

def bwr(n):
    return np.interp(np.linspace(1, 3, n), [1, 2, 3], [[0, 0, 1], [1, 1, 1], [1, 0, 0]])

def high_symmetry_points():
    # Define high symmetry points
    Gamma = np.array([0, 0])
    M = np.array([0, (2 * (1 / 3) * (np.sqrt(3)))])
    K = np.array([2 * (-1/3), (2 * (1 / 3) * (np.sqrt(3)))])

    # Plot the triangle with the vertices G, M, and K
    vertices = np.array([Gamma, M, K, Gamma])  # Close the triangle by adding G at the end
    return Gamma, M, K, vertices

def plot_berry_curvature(F, KX, KY, a, Gamma, M, K, vertices):
    
    plt.figure()
    plt.title("Berry Curvature")
    plt.imshow(np.real(F), extent=(KX[0, 0] * a / (np.pi), KX[0, -1] * a / (np.pi),
                                KY[0, 0] * a / (np.pi), KY[-1, 0] * a / (np.pi)), 
            cmap='bwr', aspect='auto')
    # Plot the triangle edges
    plt.plot(vertices[:, 0], vertices[:, 1], color='black', linewidth=0.5)
    plt.text(Gamma[0], Gamma[1], 'G', color='black', fontsize=10, va='top')
    plt.text(M[0], M[1], 'M', color='black', fontsize=10, va='bottom')
    plt.text(K[0], K[1], 'K', color='black', fontsize=10, va='bottom')

    plt.xlabel('kx a/π')
    plt.ylabel('ky a/π')
    plt.colorbar()
    plt.clim(-np.max(np.abs(np.real(F))), np.max(np.abs(np.real(F))))
    plt.show()


a, c0, B1, B2 = initialize_lattice_parameters(a=242.5 * 10**(-6))

l1, l2 = initialize_hole_parameters(a, ratio_1=0.65, ratio_2=0.35)
seperation = 0 * a
# rotation_angle = np.pi / 10
rotation_angle = 0
# unit, cell, primitive_cell, memb, hole, memb_2, hole_2 = construct_unit_cell_geometry(a, l1, l2, number_of_sides=3, polygon_with_radius=True, 
#                                                                                       symmetry_seperation=seperation)
unit = one_unit_cell(n=3, a=a, a1=l1, a2=l2, radial=False)

N_sp=100
#N=np.arange(5, 100)
N=27
N_BZ=50
berry_curv_k=[]
berry_curv_k1=[]


inv_exy, xi, yi = dielectric_function(ed=9, ea=1, unit=unit, a=a, x_start=-1, x_end=0, y_start=0, y_end=0.5, tolerance=1e-8*a, N_sp=N_sp)
#xi, yi, inv_exy = dielectric_function(ed=9, ea=1, unit=unit, a=a, x_start=-0.5, x_end=0.5, y_start=-0.5, y_end=0.5, rhombus=False, tolerance=1e-8*a)
xi, yi, inv_exy, exy=specify_dielectric_function_rectangle(a, unit, N_sp)

#NG=np.arange(10, 16)
NG=np.array([5])
for ng in NG:
    print(ng)
    khi_G_Gp, khiG, M_lin, N_lin, Mp_lin, Np_lin= fourier_coefficients_reshaped(a, ng, B1, B2, xi, yi, inv_exy)
#eigenvalues, eigenvectors, dispe_1D, G, Gx, Gy, numG = eig_val_band_structure(a, ng, B1, B2, khi_G_Gp)
    print("50% done")
    verify_dielectric(khiG, M_lin, N_lin, xi, yi, B1, B2)
#print("verification done")
    numG=(2*ng+1)**2

    #for N_BZ in N:
    #print(f"{N_BZ}")
    kx, ky, KX, KY, KX_lin, KY_lin, delta_kx, delta_ky, band_index, del_S = initialize_BZ_parameters(a, numG, N_BZ, band_index=1)
    n = band_index
    #F, H, dispe = compute_berry_curvature(n, KX, KY, del_S, compute_eigenstates_and_eigenfrequencies, N_BZ, Mp_lin, Np_lin, B1, B2, khi_G_Gp, a, numG)
    #dispe_reshaped = dispe.reshape((numG, 2 * N_BZ, 2 * N_BZ), order='C')
    Gamma, M, K, vertices = high_symmetry_points()
#plot_band_structure(dispe_reshaped, KX, KY, a, Gamma, M, K, vertices)
#plot_3D_Band_Structure(dispe_reshaped)
    #plot_berry_curvature(F, KX, KY, a, Gamma, M, K, vertices)

    k1, k2=K *np.pi/a
    k3, k4=(-K[0]*np.pi/a, K[1]*np.pi/a)
    F1, H1, dispe1, i1, j1=compute_berry_curvature_k(n, KX, KY, k1, k2, del_S, compute_eigenstates_and_eigenfrequencies, N_BZ,  Mp_lin, Np_lin, B1, B2, khi_G_Gp, a, numG)
    F2, H2, dispe2, i2, j2=compute_berry_curvature_k(n, KX, KY, k3, k4, del_S, compute_eigenstates_and_eigenfrequencies, N_BZ,  Mp_lin, Np_lin, B1, B2, khi_G_Gp, a, numG)
    #print(k1, k2, ':', KX[j1, i1], KY[j1, i1])
    #print(k3, k4, ':', KX[j2, i2], KY[j2, i2])
    
    berry_curv_k.append(F1[j1, i1])
    berry_curv_k1.append(F2[j2, i2])
    print(F1[j1, i1], F2[j2, i2])

berry_curv_k=np.array(berry_curv_k)
berry_curv_k1=np.array(berry_curv_k1)   
plt.plot(NG, berry_curv_k+berry_curv_k1, c='blue', label='K\'')
plt.ylabel('Berry curvatures')
plt.xlabel('k-mesh grid division')
#plt.plot(NG, berry_curv_k1, c='red', label='K')
plt.legend()
plt.show()

print(berry_curv_k)
print(berry_curv_k1)


