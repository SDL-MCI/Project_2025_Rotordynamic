from functions.rotorsystem import RotorSystem, sensitivity_analysis, plot_sensitivity_bars
from functions.plot import ModePlotter, compute_campbell_diagram, plot_campbell_diagram
from functions.optimization import run_optimization
import numpy as np

# === Custom Bounds Defined in Main ===
custom_bounds = [
    (0.95, 1.15),          # Beam length [m]
    (0.017, 0.017),        # Beam D_outer [m]
    (0.06, 0.1),           # Disc radius [m]
    (0.012, 0.030),        # Disc thickness [m]
    (4-1, 20-1),                # Disc pos 1 [node]
    (15-1, 20-1),              # Disc pos 2 [node]
    (3-1, 5-1),                # Bearing pos 1 [node]
    (14-1, 15-1),              # Bearing pos 2 [node]
]


# Starting point (must be within bounds)
x0 = [1.1, 0.017, 0.177/2, 0.016, 4-1, 15-1, 3-1, 14-1]  

# Ensure x0 is a 2D array with shape (popsize, len(x0))
popsize = 15
init_pop = np.random.uniform(
    low=[b[0] for b in custom_bounds],
    high=[b[1] for b in custom_bounds],
    size=(popsize, len(custom_bounds))
)

# Overwrite first individual with your starting point
init_pop[0] = x0

# === Run Optimization with Custom Bounds ===
beam, discs, bearings, Omega, result = run_optimization(custom_bounds, init_pop)


# === Setup Rotor System ===
rotor = RotorSystem(beam, discs, bearings, Omega)
rotor.assemble_global_matrices()
rotor.apply_boundary_conditions()
rotor.solve_eigenproblem()


# === Show Frequencies ===
frequencies = rotor.get_frequencies()
print("\n=== Natural Frequencies (Hz) ===")
for i, f in enumerate(frequencies[:16]):
    print(f"Mode {i+1}: {f:.2f} Hz")


# === Print Beam Parameters ===
print("\n=== Beam Parameters ===")
print(f"Length         : {beam['length']:.4f} m")
print(f"Outer Diameter : {beam['D_outer']:.4f} m")
print(f"Density        : {beam['density']} kg/m^3")
print(f"E-Modulus      : {beam['E']:.2e} Pa")
print(f"Elements       : {beam['n_elem']}")

print("\n=== Disc Positions and Geometry ===")
for i, disc in enumerate(discs, start=1):
    print(f"Disc {i}: Node {disc['pos']}, Diameter {disc['diameter']:.4f} m, Thickness {disc['thickness']:.4f} m")

print("\n=== Bearing Positions ===")
for i, b in enumerate(bearings, start=1):
    print(f"Bearing {i}: Node {b['pos']}")

print(f"\n=== Rotational Speed ===\nOmega: {Omega:.2f} rad/s ({Omega * 60 / (2 * np.pi):.2f} rpm)")


# === Plot Mode Shapes ===
plotter = ModePlotter(rotor)
plotter.plot_2D_modes(n_modes=6)
plotter.plot_3D_modes(n_modes=6)



# === Generate and Plot Campbell Diagram ===
rpm_range = np.arange(0, 1801, 50)
campbell_data = compute_campbell_diagram(rotor, rpm_range, n_modes=12)
plot_campbell_diagram(rpm_range, campbell_data)
"""
1× speed	Shaft's rotational frequency (in Hz)
2× speed	First harmonic (twice the rotational freq.)
"""


# # === Sensitivity Analysis Results and Plot ===

# param_defs = [
#     ("Beam Length", lambda r: r.L, lambda r, v: setattr(r, 'L', v)),
#     ("Beam Diameter", lambda r: r.Douter, lambda r, v: setattr(r, 'Douter', v)),
#     ("Disc Radius", lambda r: r.discs[0]['diameter']/2, lambda r, v: r.discs[0].update({'diameter': 2*v})),
#     ("Disc Thickness", lambda r: r.discs[0]['thickness'], lambda r, v: r.discs[0].update({'thickness': v})),
#     ("Disc Position", lambda r: r.discs[0]['pos'], lambda r, v: r.discs[0].update({'pos': int(v)})),
#     ("Bearing 1 Pos", lambda r: r.bearings[0]['pos'], lambda r, v: r.bearings[0].update({'pos': int(v)}))
# ]

# sensitivity = sensitivity_analysis(rotor, param_defs, delta=0.05, n_modes=6)

# print("\n=== Sensitivity Results ===")
# for name, sens in sensitivity:
#     print(f"\n{name}:")
#     for i, val in enumerate(sens):
#         print(f"  Mode {i+1}: {val:.3f} Hz / % change")

# plot_sensitivity_bars(sensitivity)
