import numpy as np
import matplotlib.pyplot as plt
import tidy3d as td
from shapely.geometry import Polygon, Point, MultiPolygon
from shapely.affinity import translate, rotate
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
    B1=2*np.pi/a/30 *np.array([1, 1/np.sqrt(3)])
    B2=2*np.pi/a/30 *np.array([1, -1/np.sqrt(3)])
    
    return a, c0, B1, B2

def initialize_hole_parameters(a, ratio_1, ratio_2):
    l1=a*ratio_1
    l2=a*ratio_2
    
    return l1, l2

def one_unit_cell (n0, n, a, a1, a2, x_centre_1=0, y_centre_1=0, x_centre_2=0, y_centre_2=0, rotation_angle_1=0, rotational_angle_2=np.pi, radial=False, symmetry_seperation=0):
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
    
    unit = translate(polygon_1, -a/2, a/(2*np.sqrt(3))).union(translate(polygon_2, 0, a/(np.sqrt(3))))
    unit1=unit
    for i in range(1, n0):
        unit=unit.union(translate(unit1, (i%2)*a/2, i*a*np.sqrt(3)/2))

    poly_1=make_regular_polygon(n=n, x_centre=x_centre_1, y_centre=y_centre_1, radial=radial, radial_distance=r1, side_length=l1, rotation_angle=rotational_angle_2)
    poly_2=make_regular_polygon(n=n, x_centre=x_centre_2, y_centre=y_centre_2, radial=radial, radial_distance=r2, side_length=l2, rotation_angle=rotation_angle_1)
    polygon_1=Polygon(poly_1)
    polygon_2=Polygon(poly_2)

    unit2 = translate(polygon_1, -a/2, -a/(2*np.sqrt(3))).union(translate(polygon_2, 0, -a/(np.sqrt(3))))
    unit3=unit2

    for i in range(1, n0):
        unit2=unit2.union(translate(unit3, (i%2)*a/2, -i*a*np.sqrt(3)/2))
    
    unit=unit.union(unit2)
    for i in range(1, n0):
        unit=unit.union(translate(unit, a*i, 0))

    #unit=unit.union(translate(unit, 2*a, 0))
    ##unit1=rotate(unit, 180)
    #unit=unit.union(translate(unit1, -a/4, -a))
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

    return inv_eps, xi, yi

def specify_dielectric_function_rectangle(a, unit, N_sp, xmin, xmax, ymin, ymax):

    ed = 9  # Relative permeability of shaded region (dielectric)
    ea = 1  # Relative permeability of white space (air)
    n1 = np.linspace(xmin, xmax, 2*N_sp, endpoint=False)
    n2 = np.linspace(ymin, ymax, 2*N_sp, endpoint=False)
  
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
    
    plt.close()
    plt.figure(figsize=(8, 6))
    plt.axes().set_aspect(0.5)
    plt.scatter(xi, yi, c=inv_exy, s=1, cmap='jet')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.colorbar(label=r'$\epsilon(x, y)$')
    plt.title('Spatial Dielectric Distribution')
    plt.show()
    
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

def eig_val_band_structure(a, numG, B1, B2, chi_p):
    #high symmetry points
    G = np.array([0, 0])
    M = np.array([0, (2 * np.pi / a * (1 / 3) * (np.sqrt(3)))])
    K = np.array([2 * np.pi / a * (-1/3), (2 * np.pi / a * (1 / 3) * (np.sqrt(3)))])
    N1=N2=N3=100
    
    #kx = np.concatenate([np.linspace(G[0], M[0], N1, endpoint=False), np.linspace(M[0], K[0], N2, endpoint=False),
     #                   np.linspace(K[0], G[0], N3)])
    #ky = np.concatenate([np.linspace(G[1], M[1], N1, endpoint=False), np.linspace(M[1], K[1], N2, endpoint=False),
     #                   np.linspace(K[1], G[1], N3)])
    X=np.array([2*np.pi/a, 0])
    kx = np.concatenate([np.linspace(-X[0], G[0], N1, endpoint=False), np.linspace(G[0], X[0], N1, endpoint=True)])
    ky = np.concatenate([np.linspace(-X[0], G[0], N1, endpoint=False), np.linspace(G[1], X[1], N1, endpoint=True)])
    
    Gx = np.array([])
    Gy = np.array([]) 
    for i in range(-numG,numG+1):
        for j in range(-numG,numG+1):
            Gx = np.append(Gx,(i*B1[0]+j*B2[0]))
            Gy = np.append(Gy,(i*B1[1]+j*B2[1]))
            #Gy=np.append(Gy, 0)
    
    G = np.array([Gx,Gy]).T
    numG = len(G)

    
    # Precompute the G matrix components for easier access and broadcasting
    Gx = G[:, 0].reshape(1, numG)  
    Gy = G[:, 1].reshape(1, numG)
    
    # Expand kx and ky for broadcasting with G
    kx_expanded = kx[:, np.newaxis]  # Shape (len(kx), 1)
    ky_expanded = ky[:, np.newaxis]  # Shape (len(kx), 1)
    
    # Precompute the components of the matrix multiplication
    kx_term = kx_expanded + Gx  # Shape (len(kx), numG)
    ky_term = ky_expanded + Gy  # Shape (len(kx), numG)
    
    kx_term_outer = kx_term[:, :, np.newaxis] * kx_term[:, np.newaxis, :]  # Shape (len(kx), numG, numG)
    ky_term_outer = ky_term[:, :, np.newaxis] * ky_term[:, np.newaxis, :]  # Shape (len(kx), numG, numG)

    # Combine the kx and ky terms
    combined_terms = kx_term_outer + ky_term_outer  # Shape (len(kx), numG, numG)

    # Finally, compute the M matrix using broadcasting and element-wise multiplication
    M = chi_p[np.newaxis, :, :] * combined_terms  # Shape (len(kx), numG, numG)
                
    # Eigen-states computation
    dispe = np.zeros((numG, len(kx)))
    for countK in range(len(kx)):
        MM = M[countK, :, :]
        eigenvalues, eigenvectors = np.linalg.eig(MM)
        dispe[:, countK] = np.sqrt(np.sort(np.real(eigenvalues)))*a/2/np.pi

    # Plotting the band structure
    plt.figure()
    ax1 = plt.gca()
    for u in range(7):
        #plt.scatter(range(len(dispe[u, :])), np.abs(dispe[u, :]), c='r', s=1)
        plt.plot(np.abs(dispe[u, :]), color='red')
        '''
        if min(dispe[u + 1, :]) > max(dispe[u, :]):
            rect_height = min(dispe[u + 1, :]) - max(dispe[u, :])
            rect = Rectangle((0, max(dispe[u, :])), N1+N2, rect_height, facecolor='lightblue')
            ax1.add_patch(rect)
        '''
    # gap_wid=0.2894240998827911-0.2616497718252288
    # gap=Rectangle((0, 0.2616497718252288), N1+N2, gap_wid, facecolor='lightblue')
    # ax1.add_patch(gap)
    # Labeling the axes+
    plt.title('Band Structure')
    plt.xticks([0, 200/3, 400/3], ['G', 'M', 'K'])
    # plt.xticks([2*np.pi/3, 4*np.pi/3], ['K\'', 'K'])
    plt.ylabel('Frequency ωa/2πc', fontsize=16)
    plt.xlabel('Wavevector', fontsize=16)
    #plt.ylim([0, 0.7])
    plt.xlim([0, N1+N2])
    plt.grid(True)
    plt.show()
    return eigenvalues, eigenvectors, dispe, G, Gx, Gy, numG


a, c0, B1, B2 = initialize_lattice_parameters(a=242.5 * 10**(-6))
ng=3
n0=15
l1, l2 = initialize_hole_parameters(a, ratio_1=0.65, ratio_2=0.35)
seperation = 0 * a
# rotation_angle = np.pi / 10
rotation_angle = 0
# unit, cell, primitive_cell, memb, hole, memb_2, hole_2 = construct_unit_cell_geometry(a, l1, l2, number_of_sides=3, polygon_with_radius=True, 
#                                                                                       symmetry_seperation=seperation)
unit = one_unit_cell(n0, n=3, a=a, a1=l2, a2=l1, radial=False)
#xi, yi, inv_exy = dielectric_function(ed=9, ea=1, unit=unit, a=a, x_start=-1, x_end=0, y_start=0, y_end=0.5, tolerance=1e-8*a, N_sp=100)
#xi, yi, inv_exy = dielectric_function(ed=9, ea=1, unit=unit, a=a, x_start=0, x_end=1, y_start=-n0/2, y_end=n0/2, rhombus=False, tolerance=1e-8*a, N_sp=200)
xi, yi, inv_exy, eps=specify_dielectric_function_rectangle(a, unit, N_sp=100, xmin=0, xmax=1, ymin=-n0/2, ymax=n0/2)
khi_G_Gp, khiG, M_lin, N_lin, Mp_lin, Np_lin= fourier_coefficients_reshaped(a, ng, B1, B2, xi, yi, inv_exy)
eigenvalues, eigenvectors, dispe_1D, G, Gx, Gy, numG = eig_val_band_structure(a, ng, B1, B2, khi_G_Gp)
