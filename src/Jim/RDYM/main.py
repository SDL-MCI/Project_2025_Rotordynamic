from functions.rotorsystem import RotorSystem
from functions.plot import ModePlotter, compute_campbell_diagram, plot_campbell_diagram
import numpy as np
# === USER INPUTS ===

beam = {
    'length': 1.1,        # [m]
    'D_outer': 0.017,        # [m]
    'density': 2700,       # [kg/m^3]
    'E': 0.7e11,          # [Pa]
    'n_elem': 20             # number of beam elements
}

discs = [
    {'pos': 6-1, 'diameter': 0.177, 'thickness': 0.016, 'density': 2700},
    {'pos': 15-1, 'diameter': 0.177, 'thickness': 0.016, 'density': 2700}
]

bearings = [
    {'pos': 3-1},
    {'pos': 14-1}
]

Omega = 1500/60*2*np.pi  # [rad/s] — optional: set to rotation speed

# === EIGENFREQUENCY ANALYSIS ===

rotor = RotorSystem(beam, discs, bearings, Omega)
rotor.assemble_global_matrices()
rotor.apply_boundary_conditions()
rotor.solve_eigenproblem()

# === PRINT RESULTS ===

frequencies = rotor.get_frequencies()
print("\n=== Natural Frequencies (Hz) ===")
for i, f in enumerate(frequencies[:16]):
    print(f"Mode {i+1}: {f:.2f} Hz")

# === PLOT RESULTS ===

plotter = ModePlotter(rotor)
plotter.plot_2D_modes(n_modes=6)
plotter.plot_3D_modes(n_modes=6)  # Optional: 3D plot if needed

# === Generate and Plot Campbell Diagram ===
rpm_range = np.arange(0, 1801, 50)
campbell_data = compute_campbell_diagram(rotor, rpm_range, n_modes=12)
plot_campbell_diagram(rpm_range, campbell_data)
"""
1× speed	Shaft's rotational frequency (in Hz)
2× speed	First harmonic (twice the rotational freq.)
"""
