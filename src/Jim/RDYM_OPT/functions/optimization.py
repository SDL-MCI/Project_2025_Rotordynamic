import numpy as np
from scipy.optimize import differential_evolution
from rotorsystem import RotorSystem
import matplotlib.pyplot as plt

# === Parameter Bounds ===
bounds = [
    (0.9, 1.2),            # Beam length [m]
    (0.05, 0.16),           # Beam D_outer [m]
    (0.06, 0.12),          # Disc radius [m]
    (0.01, 0.05),           # Disc thickness [m]
    (5, 8),               # Disc pos 1 [node index]
    (14, 17),               # Disc pos 2 [node index]
    (5, 8),               # Bearing pos 1 [node index]
    (14, 17),               # Bearing pos 2 [node index]
    (100, 2000)            # Rotation speed [rpm]
]

def objective(x):
    length, D_outer, radius, thickness, d1, d2, b1, b2, rpm = x
    density = 7833.4
    E = 2.0684e11
    n_elem = 20

    # Integer positions
    d1 = int(np.clip(round(d1), 0, n_elem))
    d2 = int(np.clip(round(d2), 0, n_elem))
    b1 = int(np.clip(round(b1), 0, n_elem))
    b2 = int(np.clip(round(b2), 0, n_elem))

    if len(set([d1, d2, b1, b2])) < 4:
        return 1e6

    discs = [
        {'pos': d1, 'diameter': 2*radius, 'thickness': thickness, 'density': density},
        {'pos': d2, 'diameter': 2*radius, 'thickness': thickness, 'density': density}
    ]
    bearings = [{'pos': b1}, {'pos': b2}]
    beam = {'length': length, 'D_outer': D_outer, 'density': density, 'E': E, 'n_elem': n_elem}
    Omega = rpm / 60 * 2 * np.pi

    try:
        rotor = RotorSystem(beam, discs, bearings, Omega)
        rotor.assemble_global_matrices()
        rotor.apply_boundary_conditions()
        rotor.solve_eigenproblem()
        freqs = rotor.get_frequencies()


        freqs = freqs[:15]  # First 3 groups (4x3)

        # Gruppierung
        group1 = freqs[0:3]
        group2 = freqs[4:7]
        group3 = freqs[8:11]
        group4 = freqs[12:15]


        # Streuung innerhalb der Gruppen
        spread1 = np.std(group1)
        spread2 = np.std(group2)
        spread3 = np.std(group3)
        spreap4 = np.std(group4)

        # Mittelwerte je Gruppe
        mean1 = np.mean(group1)
        mean2 = np.mean(group2)
        mean3 = np.mean(group3)
        mean4 = np.mean(group4)

        # Trennung zwischen Gruppen
        delta12 = mean2 - mean1
        delta23 = mean3 - mean2
        delta34 = mean4 - mean3

        # Ziel: kleine Streuung, klare Trennung, tiefe Frequenzen
        cost = (
            3 * (spread1 + spread2 + spread3 + spreap4)  # Penalize spread within groups
            - 0.5 * (delta12 + delta23 + delta34)           # Reward separation between groups
            + 10 * mean4                                  # Prefer low frequencies
        )

        return cost

    except Exception as e:
        return 1e6  # fail-safe

"""
# === Objective Function ===
def objective(x):
    length, D_outer, radius, thickness, d1, d2, b1, b2, rpm = x
    density = 7833.4   # fixed
    E = 2.0684e11
    n_elem = 20

    # Round all positions to nearest valid node
    d1 = int(np.clip(round(d1), 0, n_elem))
    d2 = int(np.clip(round(d2), 0, n_elem))
    b1 = int(np.clip(round(b1), 0, n_elem))
    b2 = int(np.clip(round(b2), 0, n_elem))

    # Constraint: All positions must be distinct
    if len(set([d1, d2, b1, b2])) < 4:
        return 1e6  # heavy penalty

    discs = [
        {'pos': d1, 'diameter': 2*radius, 'thickness': thickness, 'density': density},
        {'pos': d2, 'diameter': 2*radius, 'thickness': thickness, 'density': density}
    ]

    bearings = [
        {'pos': b1},
        {'pos': b2}
    ]

    beam = {
        'length': length,
        'D_outer': D_outer,
        'density': density,
        'E': E,
        'n_elem': n_elem
    }

    Omega = rpm / 60 * 2 * np.pi  # convert to rad/s

    try:
        rotor = RotorSystem(beam, discs, bearings, Omega)
        rotor.assemble_global_matrices()
        rotor.apply_boundary_conditions()
        rotor.solve_eigenproblem()
        freqs = rotor.get_frequencies()

        if len(freqs) < 2:
            return 1e6  # penalty if system is unstable

        f1 = freqs[0]
        spread = freqs[1] - freqs[0]

        return f1 - 0.5 * spread  # Low freq but good spread
    except Exception as e:
        return 1e6  # fail-safe penalty
"""

# === Run Optimization ===
# result = differential_evolution(objective, bounds, strategy='best1bin', maxiter=1000, popsize=15, disp=True)
result = differential_evolution(
    objective,
    bounds,
    strategy='randtobest1bin',
    maxiter=100,
    popsize=20,
    tol=1e-6,
    mutation=(0.5, 1.0),
    recombination=0.9,
    seed=42,
    disp=True
)



# === Print Result ===
print("\n=== Optimization Result ===")
labels = [
    "Beam length [m]", "Beam D_outer [m]", "Disc radius [m]", "Disc thickness [m]",
    "Disc pos 1 [node]", "Disc pos 2 [node]", "Bearing pos 1 [node]", "Bearing pos 2 [node]", "Rotation speed [rpm]"
]
for label, val in zip(labels, result.x):
    print(f"{label}: {val:.4f}")

# Print integer node positions clearly
d1, d2, b1, b2 = [int(round(result.x[i])) for i in [4, 5, 6, 7]]
print(f"\n>>> Final Node Positions (rounded):")
print(f"Disc 1 at node {d1}")
print(f"Disc 2 at node {d2}")
print(f"Bearing 1 at node {b1}")
print(f"Bearing 2 at node {b2}")

print(f"\nObjective function value: {result.fun:.4f}")





# === Frequenzen mit besten Parametern berechnen ===
# Parameter extrahieren
length, D_outer, radius, thickness, d1_, d2_, b1_, b2_, rpm = result.x
density = 7833.4
E = 2.0684e11
n_elem = 20

# Knoten runden
d1 = int(np.clip(round(d1_), 0, n_elem))
d2 = int(np.clip(round(d2_), 0, n_elem))
b1 = int(np.clip(round(b1_), 0, n_elem))
b2 = int(np.clip(round(b2_), 0, n_elem))

# Rotor neu aufbauen
discs = [
    {'pos': d1, 'diameter': 2 * radius, 'thickness': thickness, 'density': density},
    {'pos': d2, 'diameter': 2 * radius, 'thickness': thickness, 'density': density}
]
bearings = [{'pos': b1}, {'pos': b2}]
beam = {'length': length, 'D_outer': D_outer, 'density': density, 'E': E, 'n_elem': n_elem}
Omega = rpm / 60 * 2 * np.pi

# Simulation durchführen
rotor = RotorSystem(beam, discs, bearings, Omega)
rotor.assemble_global_matrices()
rotor.apply_boundary_conditions()
rotor.solve_eigenproblem()
freqs = rotor.get_frequencies()
freqs_16 = freqs[:16]

# Frequenzen ausgeben
print("\n=== Erste 16 Eigenfrequenzen [Hz] ===")
for i, f in enumerate(freqs_16, start=1):
    print(f"Mode {i:2d}: {f:.2f} Hz")