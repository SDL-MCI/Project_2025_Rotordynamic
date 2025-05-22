from functions.rotorsystem import RotorSystem
from functions.plot import ModePlotter
# === USER INPUTS ===

beam = {
    'length': 2.4384,        # [m]
    'D_inner': 0.0250,       # [m]
    'D_outer': 0.102,        # [m]
    'density': 7833.4,       # [kg/m^3]
    'E': 2.0684e11,          # [Pa]
    'n_elem': 24             # number of beam elements
}

discs = [
    {'pos': 10, 'diameter': 0.1047, 'thickness': 0.04474, 'density': 7833.4},
    {'pos': 13, 'diameter': 0.1047, 'thickness': 0.04474, 'density': 7833.4},
    {'pos': 16, 'diameter': 0.1047, 'thickness': 0.04474, 'density': 7833.4}
]

bearings = [
    {'pos': 4},
    {'pos': 22}
]

Omega = 0 #1000/60*2*np.pi  # [rad/s] — optional: set to rotation speed

# === EIGENFREQUENCY ANALYSIS ===

rotor = RotorSystem(beam, discs, bearings, Omega)
rotor.assemble_global_matrices()
rotor.apply_boundary_conditions()
rotor.solve_eigenproblem()

# === PRINT RESULTS ===

frequencies = rotor.get_frequencies()
print("\n=== Natural Frequencies (Hz) ===")
for i, f in enumerate(frequencies[:12]):
    print(f"Mode {i+1}: {f:.2f} Hz")

# === PLOT RESULTS ===

plotter = ModePlotter(rotor)
plotter.plot_2D_modes(n_modes=6)
# plotter.plot_3D_modes(n_modes=3)  # Optional: 3D plot if needed
