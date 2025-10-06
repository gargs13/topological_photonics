# -*- coding: utf-8 -*-
"""
Created on Sat Aug 23 11:21:11 2025

@author: Gargi Joshi
"""
'''
#imports
import numpy as np
import matplotlib.pyplot as plt
import tidy3d as td
import tidy3d.web as web

#parameters of waves

lambda0=0.92 #micrometers
freq0=td.C_0/lambda0 #infrared waves

lambdas=np.linspace(0.84, 1.0, 101) #wavelengths of source.... IR
freqs=td.C_0/lambdas
fwidth=0.5*(np.max(freqs)-np.min(freqs))####################
#we need fwidth to define the gaussian pulse that will be emitted from the dipole source

#parameters for crystal

s=0.14
a0=0.445 #proportional to atoms btwn 2 unit cells... can be angle or distance
t=0.16 
n_x=20 #number of unit cells in x directon
n_y=16 #number of unit cells in y direction
r=a0/3 #radial distance of triangle's centroid from lattice centre

#unit cell
def unit_cell(s, t, r, x0, y0) ->td.GeometryGroup:
    
 
    s: float (edge of triangles)
    h: float (thickness of slab)
    r: float (radial distance of triangle's centroid from lattice centre)
    x0, y0: floatss (lattice centre)

     
    coords=np.array[(s/2, -s/(2*np.sqrt(3))), (-s/2, -s/(2*np.sqrt(3))), (0, s/np.sqrt(3))]
    angles=np.arange(0, 2*np.pi, np.pi/3)
    coords_all=[]
    
    for theta in angles:
        R=np.array([[np.cos(theta+np.pi/2), -np.sin(theta+np.pi/2)], [np.sin(theta+np.pi/2), np.cos(theta+np.pi/2)]])
        rot_coords=np.dot(R, coords)
        centroid=np.array([r*np.cos(theta), r*np.sin(theta)])
        final_coords=centroid+rot_coords+np.array([x0, y0])
        
        coords_all.append(td.PolySlab(
            vertices=final_coords,
            axis=2,
            slab_bounds=(-t/2,t/2),
           ))
        
    coords_all=np.array(coords_all)
    
    return td.GeometryGroup(geometry=coords_all)

#lattice
def lattice(n_x, n_y, a0) -> list[td.GeometryGroup]:
    lattices_centers=[]
    for i in range(n_x):
        for j in range(n_y):
            x0=a0*(n_x+0.5*(j%2))
            y0=a0*(np.sqrt(3)*j/2)
            
            lattices_centers.append((x0,y0))
    
    return (lattices_centers)

#make the lattice sites
lattice_sites=lattice(n_x, n_y, a0)
#sort it by vertical spacing, so that we can divide it among top and bottom lattices
lattice_sites=sorted(lattice_sites, key=lambda t: t[1])
#division of lattices
mid=len(lattice_sites)//2

#top and bottom
top_sites=lattice_sites[:mid]
bottom_sites=lattice_sites[mid:]

#divide it in seperate packs, so that we can plot it
x_bottom, y_bottom= zip(*bottom_sites)
x_top, y_top=zip(*top_sites)

############
fig, ax=plt.subplots(1)

#top lattice triangles
top_geometry=0
for geo in top_sites:
    top_geometry+=unit_cell(s, t, r, x0=geo[0], y0=geo[1])
    
#bottom
bottom_geometry=0
for geo in bottom_sites:
    bottom_geometry+=unit_cell(s, t, r, x0=geo[0], y0=geo[1])

#holes
hole_geometry=top_geometry+bottom_geometry

#plot
bottom_geometry.plot(z=0, ax=ax, facecolor="red")
top_geometry.plot(z=0, ax=ax, facecolor="green")

ax.set_xlim((2,4))
ax.set_ylim((2,4))


GaAs=td.material_library["GaAs"]["Palik_Lossy"]

#structure details
slab=td.Structure(geometry=td.Box(center=(0,0,0),size=(td.inf, td.inf, t)), medium=GaAs)
holes=td.Structure(geometry=hole_geometry, medium=td.Medium())

#dipole
dipole_x=(n_x//2-1/2)*a0
dipole_y=(n_y//2-1)*a0*np.sqrt(3)/2

#dipole polarize in x direction
dipole_x_source=td.PointDipole(
    center=(dipole_x, dipole_y), 
    source_time=td.GaussianPulse(freq0, fwidth),
    polarization="Ex", #type of field and direction
)

dipole_y_source=td.PointDipole(
    center=(dipole_x, dipole_y), 
    source_time=td.GaussianPulse(freq0, fwidth, phase=np.pi/2), 
    #phase: -pi/2 for LCP pi/2 for RCP
    polarization="Ey",
)

#monitors
#WHY FREQ0....?
field_monitor = td.FieldMonitor(size=(td.inf, td.inf, 0), freqs=[freq0], name="field")

#size: defines the extent of the monitor box along each axis
#(0, 4, 4) -> 0: no extent along x axis, thus the monitors are in y-z plane
#perpendicular to x axis, since the interface is parallel to x axis

flux_monitor_left = td.FluxMonitor(
    size=(0, 4, 4), center=(n_x * a0 / 2 - 3, 2.6, 0), freqs=freqs, name="flux_left"
)

flux_monitor_right = td.FluxMonitor(
    size=(0, 4, 4), center=(n_x * a0 / 2 + 3, 2.6, 0), freqs=freqs, name="flux_right"
)

buffer = 0.8 * lambda0

Why buffer?
PML is designed to absorb outgoing waves without reflection
But if your slab (or sources/monitors) are too close to the PML, the near-field evanescent fields 
(or even some strong scattered fields) may directly touch the PML.
That causes artificial reflections back into your simulation... contaminating your results.
The buffer ensures that by the time the wave reaches the PML, it’s more like a free-space traveling wave 
(which the PML absorbs perfectly).
Stronger the field, more the buffer is required

    
sim=td.Simulation(
    centre=dipole_x_source.center,
    size=((n_x - 4) * a0, (n_y - 4) * np.sqrt(3) * a0 / 2, t + 2 * buffer), #shrinks domain of simulation by removing 2 unit cells from each side
    grid_spec=td.GridSpec.auto(min_steps_per_wvl=20),
    #is a way to automatically set up the spatial discretization (grid resolution) 
   #for your simulation.
   #You don’t need to manually define dx, dy, dz everywhere.
   #Use at least 20 grid points per wavelength inside the highest-index material.
   #dx<=lambda0/20
    run_time=2e-12,
    structures=[slab, holes],
    sources=[dipole_x_source, dipole_y_source],
    monitors=[field_monitor, flux_monitor_left, flux_monitor_right],
    symmetry=(0, 0, 1), #across z axis
    boundary_spec=td.BoundarySpec(
        x=td.Boundary.absorber(num_layers=60), #absorbing boundaries to avoid reflection
        y=td.Boundary.absorber(num_layers=60),
        z=td.Boundary.pml(), #to absorb outgoing waves in z direction
    ),
)
   
sim.plot(z=0)
td.web.configure(apikey="Z2FyDpnN4QHxtxt8rVZoyRmabg6x15bdMyANuJW4fXnVwgjJ")
sim_data = web.run(simulation=sim, task_name="RCP") #rcp=right circular polarized

sim_data.plot_field(
    field_monitor_name="field", field_name="E", val="abs^2", eps_alpha=0.2, vmax=5e6
)

flux_left = np.abs(sim_data["flux_left"].flux)
flux_right = np.abs(sim_data["flux_right"].flux)

fig2, ax2=plt.subplots(1)
plt.plot(lambdas, flux_right / flux_left, color="red")
plt.xlabel("Wavelength (μm)")
plt.ylabel("Flux ratio")
ax.set_xlim((0,2))
ax.set_ylim((0,25))
plt.grid(True)
plt.show()
    
'''

# -*- coding: utf-8 -*-
"""
Created on Sat Aug 23 11:21:11 2025

@author: Gargi Joshi
"""

# imports
import numpy as np
import matplotlib.pyplot as plt
import tidy3d as td
import tidy3d.web as web

# --------------------------------------------------------------------------------------
# Wave / source parameters  (units consistent with Tidy3D: length in μm)
# --------------------------------------------------------------------------------------
lambda0 = 0.92  # μm, design wavelength (IR)
freq0 = td.C_0 / lambda0  # center frequency of Gaussian pulse

lambdas = np.linspace(0.84, 1.00, 101)  # μm
freqs = td.C_0 / lambdas
fwidth = 0.5 * (np.max(freqs) - np.min(freqs))  # bandwidth for the Gaussian pulse

# --------------------------------------------------------------------------------------
# Photonic crystal parameters
# --------------------------------------------------------------------------------------
s = 0.14        # μm, edge length of each triangle
a0 = 0.445      # μm, lattice constant (center-to-center)
t = 0.16        # μm, slab thickness
n_x = 20        # number of unit cells in x
n_y = 16        # number of unit cells in y
r = a0 / 3.0    # μm, radial offset of each triangle's centroid from lattice center

# --------------------------------------------------------------------------------------
# Geometry builders
# --------------------------------------------------------------------------------------
def unit_cell(s: float, t: float, r: float, x0: float, y0: float) -> td.GeometryGroup:
    """
    Build a 6-triangle 'unit cell' (rotated copies) centered near (x0, y0).
    s: triangle edge length (μm)
    t: slab thickness (μm)
    r: radial distance of each triangle's centroid from lattice center (μm)
    x0, y0: lattice center (μm)
    """
    # Base (equilateral) triangle vertices centered at origin
    coords = np.array([
        ( s/2.0,          -s/(2.0*np.sqrt(3.0)) ),
        (-s/2.0,          -s/(2.0*np.sqrt(3.0)) ),
        ( 0.0,             s/np.sqrt(3.0)       )
    ])  # shape (3, 2)

    # 6 orientations (every 60 degrees)
    angles = np.arange(0.0, 2.0*np.pi, np.pi/3.0)

    triangles = []
    for theta in angles:
        # Rotate by theta + 90° to match your orientation choice
        rot = theta + np.pi / 2.0
        R = np.array([
            [np.cos(rot), -np.sin(rot)],
            [np.sin(rot),  np.cos(rot)]
        ])  # (2,2)

        # rotate coords: (3,2) @ (2,2) -> (3,2)
        rot_coords = coords @ R.T

        # centroid position on circle of radius r
        centroid = np.array([r*np.cos(theta), r*np.sin(theta)])

        # translate to (x0, y0)
        final_coords = rot_coords + centroid + np.array([x0, y0])

        # add a triangular PolySlab (holes: default medium later)
        triangles.append(
            td.PolySlab(
                vertices=final_coords,
                axis=2,  # z-axis extrusion
                slab_bounds=(-t/2.0, t/2.0),
            )
        )

    return td.GeometryGroup(geometry=triangles)


def lattice(n_x: int, n_y: int, a0: float):
    """
    Generate centers (x0, y0) for a hexagonal (triangular) lattice using
    row-wise staggering. Returns a list of (x0, y0) tuples.
    """
    centers = []
    for j in range(n_y):
        for i in range(n_x):
            x0 = a0 * (i + 0.5 * (j % 2))         # stagger every other row
            y0 = a0 * (np.sqrt(3.0) * j / 2.0)    # row vertical spacing
            centers.append((x0, y0))
    return centers


# --------------------------------------------------------------------------------------
# Build geometry
# --------------------------------------------------------------------------------------
# lattice sites (centers)
lattice_sites = lattice(n_x, n_y, a0)
# sort by y to split into "top" and "bottom" halves
lattice_sites = sorted(lattice_sites, key=lambda p: p[1])
mid = len(lattice_sites) // 2
top_sites = lattice_sites[:mid]
bottom_sites = lattice_sites[mid:]

# build geometry groups for each half
top_groups = [unit_cell(s, t, r, x0, y0) for (x0, y0) in top_sites]
bottom_groups = [unit_cell(s, t, r, x0, y0) for (x0, y0) in bottom_sites]

# merge into single GeometryGroup of holes
hole_geometry = td.GeometryGroup(geometry=top_groups + bottom_groups)

# quick 2D preview of just the holes (z=0 slice)
fig, ax = plt.subplots(1, figsize=(6, 6))
hole_geometry.plot(z=0.0, ax=ax)  # colored per polygon automatically
ax.set_aspect('equal', adjustable='box')
ax.set_xlabel('x (μm)')
ax.set_ylabel('y (μm)')
ax.set_title('Hole Geometry (z=0 slice)')
# adjust view to somewhere interesting if you like:
# ax.set_xlim((2, 4)); ax.set_ylim((2, 4))
plt.tight_layout()
plt.show()

# --------------------------------------------------------------------------------------
# Materials & Structures
# --------------------------------------------------------------------------------------
GaAs = td.material_library["GaAs"]["Palik_Lossy"]

slab = td.Structure(
    geometry=td.Box(center=(0.0, 0.0, 0.0), size=(td.inf, td.inf, t)),
    medium=GaAs,
)

# Air holes (default: td.Medium() ~ vacuum)
holes = td.Structure(
    geometry=hole_geometry,
    medium=td.Medium()
)

# --------------------------------------------------------------------------------------
# Sources (dipole pair for circular polarization)
# --------------------------------------------------------------------------------------
dipole_x = (n_x // 2 - 0.5) * a0
dipole_y = (n_y // 2 - 1.0) * a0 * np.sqrt(3.0) / 2.0
dipole_z = 0.0

dipole_x_source = td.PointDipole(
    center=(dipole_x, dipole_y, dipole_z),
    source_time=td.GaussianPulse(freq0=freq0, fwidth=fwidth),
    polarization="Ex",
)

dipole_y_source = td.PointDipole(
    center=(dipole_x, dipole_y, dipole_z),
    source_time=td.GaussianPulse(freq0=freq0, fwidth=fwidth, phase=np.pi/2),
    polarization="Ey",
)

# --------------------------------------------------------------------------------------
# Monitors
# --------------------------------------------------------------------------------------
# Field monitor at the central frequency only (efficient snapshot)
field_monitor = td.FieldMonitor(
    size=(td.inf, td.inf, 0.0),
    center=(dipole_x, dipole_y, 0.0),
    freqs=[freq0],
    name="field"
)

# Flux monitors sweeping full band
flux_monitor_left = td.FluxMonitor(
    size=(0.0, 4.0, 4.0),
    center=(n_x * a0 / 2.0 - 3.0, 2.6, 0.0),
    freqs=freqs,
    name="flux_left"
)

flux_monitor_right = td.FluxMonitor(
    size=(0.0, 4.0, 4.0),
    center=(n_x * a0 / 2.0 + 3.0, 2.6, 0.0),
    freqs=freqs,
    name="flux_right"
)

# --------------------------------------------------------------------------------------
# Simulation domain, grid, boundaries
# --------------------------------------------------------------------------------------
buffer = 0.8 * lambda0  # extra space to PML in z

sim = td.Simulation(
    center=(dipole_x, dipole_y, 0.0),
    size=((n_x - 4) * a0, (n_y - 4) * np.sqrt(3.0) * a0 / 2.0, t + 2.0 * buffer),
    grid_spec=td.GridSpec.auto(min_steps_per_wvl=20),
    run_time=2e-12,  # s
    structures=[slab, holes],
    sources=[dipole_x_source, dipole_y_source],
    monitors=[field_monitor, flux_monitor_left, flux_monitor_right],
    symmetry=(0, 0, 1),
    boundary_spec=td.BoundarySpec(
        x=td.Boundary.absorber(num_layers=60),
        y=td.Boundary.absorber(num_layers=60),
        z=td.Boundary.pml()
    ),
)

# quick scene check
sim.plot(z=0.0)

# --------------------------------------------------------------------------------------
# Run on web
# --------------------------------------------------------------------------------------
td.web.configure(apikey="Z2FyDpnN4QHxtxt8rVZoyRmabg6x15bdMyANuJW4fXnVwgjJ")
sim_data = web.run(simulation=sim, task_name="RCP")

# --------------------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------------------
# Field map at freq0
sim_data.plot_field(
    field_monitor_name="field", field_name="E", val="abs^2", eps_alpha=0.2, vmax=5e6
)

# Flux spectra and ratio
flux_left = np.abs(sim_data["flux_left"].flux)
flux_right = np.abs(sim_data["flux_right"].flux)

fig2, ax2 = plt.subplots(1, figsize=(6, 4))
ax2.plot(lambdas, flux_right / flux_left)
ax2.set_xlabel("Wavelength (μm)")
ax2.set_ylabel("Flux ratio (right / left)")
ax2.set_xlim((lambdas.min(), lambdas.max()))
ax2.grid(True)
plt.tight_layout()
plt.show()
