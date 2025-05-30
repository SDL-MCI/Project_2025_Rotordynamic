from functions.rotorsystem import RotorSystem
from functions.plot import ModePlotter
from functions.optimization import run_optimization
import numpy as np

# === Custom Bounds Defined in Main ===
custom_bounds = [
    (0.95, 1.15),          # Beam length [m]
    (0.005, 0.016),        # Beam D_outer [m]
    (0.08, 0.10),          # Disc radius [m]
    (0.006, 0.015),        # Disc thickness [m]
    (5, 8),                # Disc pos 1 [node]
    (14, 17),              # Disc pos 2 [node]
    (5, 8),                # Bearing pos 1 [node]
    (14, 17),              # Bearing pos 2 [node]
    (500, 1500)            # Rotation speed [rpm]
]


# === Run Optimization with Custom Bounds ===
beam, discs, bearings, Omega, result = run_optimization(custom_bounds)


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
plotter.plot_2D_modes(n_modes=3)
#plotter.plot_3D_modes(n_modes=6)