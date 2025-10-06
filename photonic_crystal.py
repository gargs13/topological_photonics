import matplotlib.pyplot as plt
import numpy as np
import tidy3d as td
import tidy3d.web as web

#parameters of waves

lda0=0.92 #wavelength, lambda 0
freq0=td.C_0/lda0 #freq corresponding to lambda 0

ldas=np.linspace(0.84, 1.0, 101) #wavelengths of source
freqs=td.C_0/ldas

fwidth=0.5*(np.max(freqs)-np.min(freqs)) #frequency width of source

#geometric parameters for photonic crystals

s=0.14 #triangle edge length
a0=0.445 #lattice constant
h=0.16 #slab thickness
Nx=20 #number of unit cells in x direction
Ny=16 #number of unit cells in y direction

#unit cell at a given center position, for different sites in lattice

#construct a unit cell geometry of equ triangles around (x0,y0)
def make_unit_cell(
        s:float, a0:float, r:float, h:float, x0:float, y0:float
    ) -> td.GeometryGroup:
    #Geometry group= A collection of Geometry objects that can be called as a single geometry object.
    '''
s: edge length of each equilateral triangle
a0: lattice constant
r: radial distance from center of unit cell to centroid of each triangle
h: thickness of slab
'''
    
    vertices_0=s*np.array(
        [
            [0.0, np.sqrt(3)/3], #top vertex
            [-0.5, -np.sqrt(3)/6], #bottom-left
            [0.5, -np.sqrt(3)/6],  #bottom-right
        ]
    )
    
      
    angles=np.arange(0, 2*np.pi, np.pi/3)  #how to place the triangles in a cirlce
    
    triangles=[]
    
    for angle in angles:
        
        R=np.array(
            [
                [np.cos(angle+np.pi/2), -np.sin(angle+np.pi/2)],
                [np.sin(angle+np.pi/2), np.cos(angle+np.pi/2)],
                
            ]
        ) #rotational matrix
    
        verts_rot=vertices_0 @ R.T #rotate the triangle about the origin
        
        centre=np.array([r*np.cos(angle) + x0, r*np.sin(angle)+y0]) #centroid of triangle
                
        verts=verts_rot+centre #translate to the correct position
        
        triangles.append(
            td.PolySlab(
                vertices=verts,
                axis=2,
                slab_bounds=(-h/2,h/2),
            )
        )
        #polyslab: Polygon extruded with optional sidewall angle along axis direction.
    return td.GeometryGroup(geometries=triangles)

r=a0/3

'''
#TEST

triangles=make_unit_cell(s, a0, r, h, x0=1, y0=2)
print(triangles)

triangles.plot(z=h/3)
plt.show()
'''

#calculate centers of lattice sites
#seperate top and bottom half 

def generate_triangle_lattice(a0:float, Nx:int, Ny:int) -> list[tuple[float,float]]:
    '''
    centre=(x,y)

    Parameters
    ----------
    a0 : float
        Lattice constant
    Nx : int
        number of sites in x direction.
    Ny : int
        number of sites in y

    Returns
    -------
    list[tuple[float,float]]
        each lattice site x:float, y:float, 
        coordinate=(x,y)
        list of all=[(x,y)]

    '''
    coords=[]
    for j in range(Ny):
        for i in range(Nx):
           x=a0*(i+0.5*(j%2)) #shift every row by a0/2 to create triangular pattern
           y=a0*(np.sqrt(3)/2)*j #vertical spacing between rows
           coords.append((x,y))
    return coords

lattice_sites=generate_triangle_lattice(a0, Nx, Ny)

lattice_sites_sorted=sorted(lattice_sites, key=lambda xy:xy[1]) #xy, are the tuples(x,y) xy[1] mtlb sort by y

N_total = len(lattice_sites_sorted) 

N_half = N_total // 2

bottom_sites = lattice_sites_sorted[:N_half]
top_sites = lattice_sites_sorted[N_half:]

x_bottom, y_bottom = zip(*bottom_sites) #zip combines x coordinates in one tuple and y coordinate in another tuple
x_top, y_top = zip(*top_sites)

'''
plt.figure(figsize=(6, 6))
plt.scatter(x_bottom, y_bottom, color="red", label="Bottom Half")
plt.scatter(x_top, y_top, color="green", label="Top Half")
plt.gca().set_aspect("equal")
plt.title("Triangular Lattice Split into Top and Bottom Halves")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()
'''

fig, ax = plt.subplots(1)

r = 0.94 * a0 / 3

lower_half_geometry = 0
for site in bottom_sites:
    lower_half_geometry += make_unit_cell(s, a0, r, h, x0=site[0], y0=site[1])

r = 1.05 * a0 / 3
top_half_geometry = 0
for site in top_sites:
    top_half_geometry += make_unit_cell(s, a0, r, h, x0=site[0], y0=site[1])

hole_geometry = lower_half_geometry + top_half_geometry
lower_half_geometry.plot(z=0, ax=ax, facecolor="red")
top_half_geometry.plot(z=0, ax=ax, facecolor="green")

ax.set_xlim((2,4))
ax.set_ylim((2,4))

#plt.plot(np.array((Nx//2-1/2)*a0), np.array((Ny//2-1)*a0*np.sqrt(3)/2), 'o')
#plt.show()

#for material property (optical), we will use the material library
#GaAs
#source: circularly polarized point dipoles (two cross-polarized dipoles with a 90-degree phase shift)
#monitor: FieldMonitor to visualize the unidirectional propagation of the edge mode: record electromagnetic field
#two FluxMonitors to measure the transmission power to the left and right: records power flux 

GaAs=td.material_library["GaAs"][ #material property
    "Palik_Lossy"
    ]

slab=td.Structure(geometry=td.Box(center=(0,0,0), size=(td.inf,td.inf, h)), medium=GaAs) #GaAs slab
holes=td.Structure(geometry=hole_geometry, medium=td.Medium()) #create air hole structures

#place the dipole source in the center of the shrunken site at the interface
dipole_pos_x=(Nx//2-1/2)*a0
dipole_pos_y=(Ny//2-1)*a0*np.sqrt(3)/2

#dipole polarized in x direction
dipole_source_x=td.PointDipole(
    center=(dipole_pos_x,dipole_pos_y, 0),
    source_time=td.GaussianPulse(freq0=freq0, fwidth=fwidth), #excitation frequency
    polarization="Ex", #Ex gives type of current and direction (x)
)

# dipole in the y direction with a pi/2 phase
dipole_source_y = td.PointDipole(
    center=(dipole_pos_x, dipole_pos_y, 0),
    source_time=td.GaussianPulse(freq0=freq0, fwidth=fwidth, phase=np.pi / 2),
    polarization="Ey",
)

field_monitor = td.FieldMonitor(size=(td.inf, td.inf, 0), freqs=[freq0], name="field")

flux_monitor_left = td.FluxMonitor(
    size=(0, 4, 4), center=(Nx * a0 / 2 - 3, 2.6, 0), freqs=freqs, name="flux_left"
)

flux_monitor_right = td.FluxMonitor(
    size=(0, 4, 4), center=(Nx * a0 / 2 + 3, 2.6, 0), freqs=freqs, name="flux_right"
)

buffer = 0.8 * lda0 #defines extra space (padding) above and below 
#the photonic slab in the z-direction, to prevent reflections from interfering with the simulation.
#ensures that the PML (perfectly matched layer) or absorbing boundary has space to absorb outgoing waves cleanly

sim = td.Simulation(
    center=dipole_source_x.center,
    size=((Nx - 4) * a0, (Ny - 4) * np.sqrt(3) * a0 / 2, h + 2 * buffer), #shrinks domain of simulation by removing 2 unit cells from each side
    grid_spec=td.GridSpec.auto(min_steps_per_wvl=20),
    run_time=2e-12,
    structures=[slab, holes],
    sources=[dipole_source_x, dipole_source_y],
    monitors=[field_monitor, flux_monitor_left, flux_monitor_right],
    symmetry=(0, 0, 1), 
    boundary_spec=td.BoundarySpec(
        x=td.Boundary.absorber(num_layers=60), #absorbing boundaries to avoid reflection
        y=td.Boundary.absorber(num_layers=60),
        z=td.Boundary.pml(), #to absorb outgoing waves in z direction
    ),
)

sim.plot(z=0)
#plt.show()

td.web.configure(apikey="Z2FyDpnN4QHxtxt8rVZoyRmabg6x15bdMyANuJW4fXnVwgjJ")
sim_data = web.run(simulation=sim, task_name="RCP") #rcp=right circular polarized


# plot field monitor data
sim_data.plot_field(
    field_monitor_name="field", field_name="E", val="abs^2", eps_alpha=0.2, vmax=5e6
)
#plt.show()

# Extract flux data from simulation results
flux_left = np.abs(sim_data["flux_left"].flux)
flux_right = np.abs(sim_data["flux_right"].flux)

fig2, ax2=plt.subplots(1)
plt.plot(ldas, flux_right / flux_left, color="red")
plt.xlabel("Wavelength (μm)")
plt.ylabel("Flux ratio")
ax.set_xlim((0,2))
ax.set_ylim((0,25))
plt.grid(True)
plt.show()



        


    















