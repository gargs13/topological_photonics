# -*- coding: utf-8 -*-
"""
Created on Tue Sep 30 00:09:28 2025

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


def initialize_lattice_parameters(a, numG):
    a=a
    numG=numG
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
    '''
    fig, ax = plt.subplots(figsize=(6, 6))
    if unit.geom_type == 'Polygon':
        x, y = unit.exterior.xy
        plt.fill(x, y, color='orange', edgecolor='black', alpha=0.6)
    elif unit.geom_type == 'MultiPolygon':
        for p in unit.geoms:
            x, y = p.exterior.xy
            plt.fill(x, y, color='orange', edgecolor='black', alpha=0.6)
    ax.set_aspect('equal', 'box')
    ax.set_title("Unit Cell Geometry")
    ax.set_xlabel("x [a.u.]")
    ax.set_ylabel("y [a.u.]")
    ax.grid(True, linestyle='--', alpha=0.4)
            
    plt.show()
    '''
    return unit


def dielectric_function(ed, ea, unit, a, x_start, x_end, y_start, y_end, tolerance=0, N_sp=100, rhombus=True):
    
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

def specify_dielectric_function_rectangle(a, unit):
    import numpy as np
    import matplotlib.pyplot as plt
    from shapely.geometry import Point
    ed = 9  # Relative permeability of shaded region (dielectric)
    ea = 1  # Relative permeability of white space (air)

    # Spatial coordinates
    N_sp = 100
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
    return xi, yi, inv_exy

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
    
    #reshaping it
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
    '''
    # Plot Chi(G-G')
    plt.figure()
    plt.imshow(np.abs(chi_p), norm=matplotlib.colors.LogNorm())
    plt.xlabel("G'")
    plt.ylabel("G")
    plt.colorbar(label=r'$\chi(G-G)$')
    plt.title("Chi(G-G') Matrix")
    plt.show()
    '''
    return chi_p, chi, M_lin, N_lin, Mp_lin, Np_lin


#1'       

def initialize_BZ_parameters(a, numG, num_BZ, band_index=1):
    num_BZ=num_BZ
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
    
      #  print(f"part2: {i / (sz[0] - 1) * 100:.2f}% completed", end='\r')
    
    F = np.imag(np.log(U)) / del_S 
    #Complex log of the Wilson loop product, gives Berry phase.
    #np.imag=>Extract the Berry curvature contribution (phase) per plaquette.
    #/delS to get curvature density
    return F, H, dispe

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
    dispe[:, i_index] = np.sqrt(np.sort(np.real(w))) * a / (2 * np.pi)

    w = np.real(w)
    IX = np.argsort(w)
    w = w[IX]
    
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

def plot_band_structure(dispe_reshaped, KX, KY, a, Gamma, M, K, vertices):
    plt.figure()
    plt.imshow(dispe_reshaped[0,:], extent=(KX[0, 0] * a / (np.pi), KX[0, -1] * a / (np.pi),
                                            KY[0, 0] * a / (np.pi), KY[-1, 0] * a / (np.pi)) 
            ,cmap = 'viridis')
    # Plot the triangle edges
    plt.plot(vertices[:, 0], vertices[:, 1], color='white', linewidth=1)
    plt.text(Gamma[0], Gamma[1], 'G', color='white', fontsize=12, va='top')
    plt.text(M[0], M[1], 'M', color='white', fontsize=12, va='bottom')
    plt.text(K[0], K[1], 'K', color='white', fontsize=12, va='bottom')

    plt.xlabel('kx a/π')
    plt.ylabel('ky a/π')
    plt.colorbar(label="Frequency ωa/2πc")
    plt.title("2D Band Structure (Lowest Band)")
    plt.show()


def plot_3D_Band_Structure(dispe_reshaped):

    Z_1 = dispe_reshaped[0, :]  #lowest band at a fixed Kz
    Z_2 = dispe_reshaped[1, :]  #second lowest band at a fixed Kz
    Z_3 = dispe_reshaped[2, :]  #third lowest band at a fixed Kz
    #meshgrid for Kx and Ky dimensions
    Kx, Ky = np.meshgrid(np.arange(Z_1.shape[1]), np.arange(Z_1.shape[0]))

    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plotting the surface
    surface_1 = ax.plot_surface(Kx, Ky, Z_1, cmap='jet', vmin=0, vmax=0.45)
    surface_2 = ax.plot_surface(Kx, Ky, Z_2, cmap='jet', vmin=0, vmax=0.45)
    surface_3 = ax.plot_surface(Kx, Ky, Z_3, cmap='jet', vmin=0, vmax=0.5)

    # Adding labels and title
    ax.set_xlabel("Kx")
    ax.set_ylabel("Ky")
    ax.set_zlabel("Frequency ωa/2πc")
    plt.title("3D Band Structure (2 Lowest Bands)")

    # Add a color bar which maps values to colors
    cbar = fig.colorbar(surface_1, ax=ax, pad=0.1)
    cbar.set_label("Frequency ωa/2πc")

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

def chern_number_calculation_polygon_area(F, KX, KY, a, del_S, vertices_1, vertices_2, kx_center_K, ky_center_K, kx_center_K_prime, ky_center_K_prime):
    def is_point_in_polygon(x, y, polygon): 
        path = Path(polygon)
        return path.contains_point((x, y))

    def calculate_chern_number(F, KX, KY, vertices): 
        indices = []
        for i in range(KX.shape[0]):
            for j in range(KX.shape[1]):
                if is_point_in_polygon(KX[i, j], KY[i, j], vertices):
                    indices.append(np.array([i, j]))
        indices = np.array(indices)

        F_within_polygon=np.array([])
        for each in indices:
            F_within_polygon=np.append(F_within_polygon, F[each[0], each[1]])
        return (1 / (2 * np.pi)) * np.sum(F_within_polygon * del_S)
    '''
    # Plotting Berry curvature
    plt.figure()
    plt.title("Berry Curvature")
    plt.imshow(np.real(F), extent=(KX[0, 0] * a / (2 * np.pi), KX[0, -1] * a / (2 * np.pi),
                                   KY[0, 0] * a / (2 * np.pi), KY[-1, 0] * a / (2 * np.pi)),
               cmap='bwr', aspect='auto')
    
    # Create a polygon patch for the region
    polygon_patch_1 = poly(vertices_1 * (a / (2 * np.pi)), closed=True, fill=False, edgecolor='red', linewidth=1)
   
    plt.gca().add_patch(polygon_patch_1)
    plt.text(kx_center_K * a / (2 * np.pi), ky_center_K * a / (2 * np.pi) + 0.05, 'K', color='red', fontsize=12)
    
    # Create a polygon patch for the region
    polygon_patch_2 = poly(vertices_2 * (a / (2 * np.pi)), closed=True, fill=False, edgecolor='blue', linewidth=1)
    plt.gca().add_patch(polygon_patch_2)
    plt.text(kx_center_K_prime * a / (2 * np.pi), ky_center_K_prime * a / (2 * np.pi) + 0.05, "K'", color='blue', fontsize=12)

    plt.xlabel('kx a/2π')
    plt.ylabel('ky a/2π')
    plt.colorbar()
    plt.clim(-np.max(np.abs(np.real(F))), np.max(np.abs(np.real(F))))
    #plt.show()
    '''
    chern_number_K = calculate_chern_number(F, KX, KY, vertices_1)
    chern_number_K_prime = calculate_chern_number(F, KX, KY, vertices_2)

    return chern_number_K, chern_number_K_prime
#

a, c0, B1, B2 = initialize_lattice_parameters(a=242.5 * 10**(-6), numG=5)
l1, l2 = initialize_hole_parameters(a, ratio_1=0.65, ratio_2=0.35)
seperation = 0 * a
# rotation_angle = np.pi / 10
rotation_angle = 0
# unit, cell, primitive_cell, memb, hole, memb_2, hole_2 = construct_unit_cell_geometry(a, l1, l2, number_of_sides=3, polygon_with_radius=True, 
#                                                                                       symmetry_seperation=seperation)
unit = one_unit_cell(n=3, a=a, a1=l1, a2=l2, radial=False)
xi, yi, inv_exy = dielectric_function(ed=9, ea=1, unit=unit, a=a, x_start=-1, x_end=0, y_start=0, y_end=0.5, tolerance=1e-8*a)
xi, yi, inv_exy = dielectric_function(ed=9, ea=1, unit=unit, a=a, x_start=-0.5, x_end=0.5, y_start=-0.5, y_end=0.5, rhombus=False, tolerance=1e-8*a)
xi, yi, inv_exy=specify_dielectric_function_rectangle(a, unit)

#N_G=np.arange(6, 8)
#N=np.arange(5, 20, 2)

#(0.24215403655920903, -0.24214048167701763) N=7
#(0.23982359313725737, -0.23964272102203918) 
ng=5
N=np.arange(2, 50)
#N=np.array([12])

chern_k=[]
chern_k1=[]
error=[]

Gamma, M, K, vertices = high_symmetry_points()
dx=0.25
dy=0.25
K=K*np.pi/a
K_prime=np.array([-K[0], K[1]])
rad=np.linalg.norm(K-Gamma)
vertices_1=make_regular_polygon(3, K[0], K[1], True, rad, 0, np.pi/2)
vertices_1=np.vstack((vertices_1, vertices_1[0]))
vertices_2=make_regular_polygon(3, K_prime[0], K_prime[1], True, rad, 0, -np.pi/2)
vertices_2=np.vstack((vertices_2, vertices_2[0]))


# for ng in N:

khi_G_Gp, khiG, M_lin, N_lin, Mp_lin, Np_lin= fourier_coefficients_reshaped(a, ng, B1, B2, xi, yi, inv_exy)
numG=(2*ng+1)**2
for N_BZ in N:
    print(N_BZ)
    kx, ky, KX, KY, KX_lin, KY_lin, delta_kx, delta_ky, band_index, del_S = initialize_BZ_parameters(a, numG, N_BZ, band_index=1)
    n = band_index
    F, H, dispe = compute_berry_curvature(n, KX, KY, del_S, compute_eigenstates_and_eigenfrequencies, N_BZ, Mp_lin, Np_lin, B1, B2, khi_G_Gp, a, numG)
    dispe_reshaped = dispe.reshape((numG, 2 * N_BZ, 2 * N_BZ), order='C')
    #plot_band_structure(dispe_reshaped, KX, KY, a, Gamma, M, K, vertices)
    #plot_3D_Band_Structure(dispe_reshaped)
    #plot_berry_curvature(F, KX, KY, a, Gamma, M, K, vertices)
    #chernk, chernk1=chern_number_calculation_rectangle_area(F, KX, KY, a, del_S, dx, dy)
    chernk, chernk1=chern_number_calculation_polygon_area(F, KX, KY, a, del_S, vertices_2, vertices_1, K[0], K[1], K_prime[0], K_prime[1])
    chern_k.append(chernk)
    chern_k1.append(chernk1)
    print(f"{chernk, chernk1}")
    error.append(chernk+chernk1)


print(chern_k)
print(chern_k1)

plt.plot(N, chern_k, color='red')
#plt.xlabel("k-mesh grid division")
plt.ylabel("chern number at K")
plt.show()

plt.plot(N, chern_k1, color='blue')
#plt.xlabel("k-mesh grid division")
plt.ylabel("chern number at K'")
plt.show()

plt.plot(N, error, color='green')
zer=np.zeros(N.size)
plt.plot(N, zer, color='black')
plt.plot()
'''
radius = 2 * np.pi / a * (2 / 3)
# Define the region of interest around the K point
kx_center_K = 2 * np.pi / a * (1 / 3)
ky_center_K = 2 * np.pi / a * (1 / np.sqrt(3))
vertices_1 = make_regular_polygon(n, x_centre=kx_center_K, y_centre=ky_center_K, radial=True, rotation_angle=-np.pi/2, radial_distance=radius)

vertices_1 = np.array(vertices_1)
vertices_1 = np.vstack((vertices_1, vertices_1[0]))

n = 3
radius = 2 * np.pi / a * (2 / 3)
# Define the region of interest around the K point
kx_center_K_prime = 2 * np.pi / a * (2 / 3)
ky_center_K_prime = 0
# ky_center_K_prime = 2 * np.pi / a * (1 / np.sqrt(3))
vertices_2 = make_regular_polygon(n, x_centre=kx_center_K_prime, y_centre=ky_center_K_prime, radial=True, rotation_angle=np.pi/2, radial_distance=radius)
vertices_2 = np.array(vertices_2)
vertices_2 = np.vstack((vertices_2, vertices_2[0]))
chern_number_calculation_polygon_area(F, KX, KY, a, del_S, vertices_1, vertices_2, kx_center_K, ky_center_K, kx_center_K_prime, ky_center_K_prime)

idx = 0.25
idy = 0.25
chern_number_calculation_rectangle_area(F, KX, KY, a, del_S, dx=idx, dy=idy)
'''