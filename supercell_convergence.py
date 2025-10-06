# -*- coding: utf-8 -*-
"""
Created on Wed Sep 24 22:21:21 2025

@author: Gargi Joshi
"""
#imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

#parameters
a=1
f=0.01 #filling ratio
epsa=1
eps_d=12
mesh_div=1000
#numG=3

#one supercell

#hole_centers
yt=a/4
hole_centers=[]

for i in range(2):
    hole_centers.append([-a/2,  yt])
    hole_centers.append([ a/2,  yt])
    hole_centers.append([-a/2, -yt])
    hole_centers.append([ a/2, -yt])
    
    yt += a * np.sqrt(3)/2

hole_centers.append([0,  (0.25+np.sqrt(3)/4)*a])
hole_centers.append([0, -(0.25+np.sqrt(3)/4)*a])

hole_centers = np.array(hole_centers)

xmax=np.max(hole_centers[:, 0])
ymax=np.max(hole_centers[:, 1])

area_cell=(2*xmax)*(2*ymax)
r=np.sqrt(area_cell*f/(5*np.pi))  

X = np.linspace(-xmax, xmax, mesh_div)
Y = np.linspace(-ymax, ymax, mesh_div)
xMesh, yMesh = np.meshgrid(X, Y)
coords = np.stack([xMesh, yMesh], axis=-1) # shape (Ny, Nx, 2)


dists = np.linalg.norm(coords[..., None, :] - hole_centers[None, None, :, :], axis=-1)
nearest = np.min(dists, axis=-1)

inv_eps = np.where(nearest < r, 1/epsa, 1/eps_d)
      
plt.close()
plt.figure(figsize=(8, 6))
#plt.scatter(xMesh, yMesh, c=inv_eps, s=1, cmap='jet')
#plt.pcolormesh(xMesh, yMesh, inv_eps, cmap='jet', shading='auto')
plt.imshow(1/inv_eps, extent=[-xmax-0.5, xmax+0.5, -ymax-0.5, ymax+0.5], origin='lower', cmap='jet', interpolation='bicubic', aspect='equal')
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.colorbar(label=r'$\epsilon(x, y)$')
plt.title('Spatial Dielectric Distribution')
plt.show()


precis=5

#precisy=precisx*ymax/xmax
kx=np.r_[
    np.linspace(0, xmax, precis+1),
    np.full(precis-1, xmax),
    np.linspace(xmax, 0, precis+1),
]

ky=np.r_[
    np.zeros(precis),
    np.linspace(0, ymax, precis+1),
    np.linspace(ymax-ymax/precis, 0, precis)    
]

fft_inv_eps = np.fft.fft2(inv_eps) / inv_eps.size
fft_inv_eps = np.fft.fftshift(fft_inv_eps)

Ny, Nx = inv_eps.shape

gx_vals = np.fft.fftshift(np.fft.fftfreq(Nx, d=(2*xmax)/Nx)) * 2*np.pi
gy_vals = np.fft.fftshift(np.fft.fftfreq(Ny, d=(2*ymax)/Ny)) * 2*np.pi

chi_dict = {}

for ix, gx in enumerate(gx_vals):
    for iy, gy in enumerate(gy_vals):
        chi_dict[(round(gx, 8), round(gy, 8))] = fft_inv_eps[iy, ix]


NG=[3]
[1, 3, 5, 7, 9, 10, 11, 13]
#NG=[3, 5, 7, 9, 11]

plt.figure()
band_index=[1, 2, 3, 4]
N_m=[]
dispe_band=[]
for numG in NG:
    G0x=np.arange(-numG*2*np.pi/a, (numG)*2*np.pi/a, 2*np.pi/a)
    G0y=np.arange(-5*numG*2*np.pi/a, 5*(numG)*2*np.pi/a, 2*np.pi/a)
    G=np.array([[gx, gy] for gx in G0x for gy in G0y])

    #A_mesh=(xmax/mesh_div)*(ymax/mesh_div)

    chi= np.zeros((len(G), len(G)), dtype=complex)

    for g in range(len(G)):
        for g1 in range(len(G)):
        
            dG = (G[g][0]-G[g1][0], G[g][1]-G[g1][1])
            chi[g, g1] = chi_dict.get((round(dG[0],8), round(dG[1],8)), 0.0)
    kvec=np.array([np.pi/a, 0])
    '''
    M=np.zeros((len(G), len(G)))

    for g in range(len(G)):
        for g1 in range(len(G)):
            M[g,g1] = chi[g,g1] * np.dot(kvec+G[g], kvec+G[g1])
    eig_val, eig_vec=np.linalg.eig(M)
    eig_val = np.abs(np.sort(np.real(eig_val)))
    dispersion = np.sqrt(eig_val) * a / (2*np.pi)
    list_bands=[]
    m=[]
    for idx in band_index:
        list_bands.append(dispersion[idx])
        m.append(numG)
   # m=np.full(dispersion.shape, numG)
    dispe_band.append(list_bands)
    N_m.append(m)
    print(numG)
   # plt.scatter(m, dispersion[], c='red', s=2)
plt.plot(N_m, dispe_band)
plt.show()
'''

M = np.zeros((len(kx), len(G), len(G)), dtype=complex)
print(0)
for ki in range(len(kx)):
    kvec = np.array([kx[ki], ky[ki]])
    for g in range(len(G)):
        for g1 in range(len(G)):
            M[ki,g,g1] = chi[g,g1] * np.dot(kvec+G[g], kvec+G[g1])

dispersion = np.zeros((len(G), len(kx)))
print(2)
for k in range(len(kx)):
    eig_val, eig_vec = np.linalg.eig(M[k])
    eig_val = np.abs(np.sort(np.real(eig_val)))
    dispersion[:, k] = np.sqrt(eig_val) * a / (2*np.pi)
print(3)
plt.figure(figsize=(8, 6))
ax1 = plt.gca()
plt.title("Band structures")
for u in range(10):
    plt.plot(dispersion[u, :])
    if min(dispersion[u + 1, :]) > max(dispersion[u, :]):
            rect_height = min(dispersion[u + 1, :]) - max(dispersion[u, :])
            rect = Rectangle((0, max(dispersion[u, :])), precis*3, rect_height, facecolor='lightblue')
            ax1.add_patch(rect)
plt.ylabel('Frequency ωa/2πc', fontsize=16)
plt.xlabel('Wavevector', fontsize=16)
plt.grid(True)
plt.show()
    


    

'''
# --- convergence study ---
numG_list = [3,5,7,9,11,13]
nbands = 3
freqs_vs_G = np.zeros((len(numG_list), nbands))

kvec = np.array([0,0])   # Gamma point
for i, nG in enumerate(numG_list):
    freqs_vs_G[i,:] = compute_bands_at_k(kvec, nG, nbands)

# plot
plt.figure(figsize=(6,5))
for b in range(nbands):
    plt.plot(numG_list, freqs_vs_G[:,b], 'o-', label=f'Band {b+1}')
plt.xlabel('numG (reciprocal truncation)')
plt.ylabel(r'$\omega a / 2\pi c$')
plt.title('Convergence of band frequencies at Γ')
plt.legend()
plt.grid(True, ls='--', alpha=0.3)
plt.show()
'''