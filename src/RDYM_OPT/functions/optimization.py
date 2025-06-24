import numpy as np
from scipy.optimize import differential_evolution
from functions.rotorsystem import RotorSystem
import matplotlib.pyplot as plt

# === Objective Function ===
def objective(x):
    length, D_outer, radius, thickness, d1, d2, b1, b2 = x
    rpm = 1500
    density = 2700
    E = 0.7e11
    n_elem = 20

    d1 = int(np.clip(round(d1), 0, n_elem))
    d2 = int(np.clip(round(d2), 0, n_elem))
    b1 = int(np.clip(round(b1), 0, n_elem))
    b2 = int(np.clip(round(b2), 0, n_elem))

    if len(set([d1, d2, b1, b2])) < 4:
        return 1e6

    discs = [
        {'pos': d1, 'diameter': 2 * radius, 'thickness': thickness, 'density': density},
        {'pos': d2, 'diameter': 2 * radius, 'thickness': thickness, 'density': density}
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

        freqs = freqs[:15]
        group1 = freqs[0:3]
        group2 = freqs[4:7]
        group3 = freqs[8:11]
        group4 = freqs[12:15]
        
        spread1 = np.std(group1)
        spread2 = np.std(group2)
        spread3 = np.std(group3)
        spread4 = np.std(group4)

        mean1 = np.mean(group1)
        mean2 = np.mean(group2)
        mean3 = np.mean(group3)
        mean4 = np.mean(group4)

        delta12 = mean2 - mean1
        delta23 = mean3 - mean2
        delta34 = mean4 - mean3

        cost = (
            -5 * (spread1 + spread2 + spread3 + spread4)   # Penalize frequency spread within groups
            - 5 * (delta12 + delta23 + delta34)           # Reward frequency separation between groups
            + 5 * mean4                                   # Penalize high frequency in group 4
            - 10 * mean1                                  # Reward high frequency in group 1
        )
        if not (28 <= mean1 <= 33):
            cost += 100  # or cost = 1e6

        if not ( mean4 < 275):
            cost += 100  # or cost = 1e6

        return cost

    except Exception:
        return 1e6  # Fallback if eigenproblem fails


# === Run Optimization with Given Bounds ===
def run_optimization(bounds, init_pop):
    result = differential_evolution(objective, bounds, strategy='best1bin', maxiter=10, popsize=15, init= init_pop,disp=True)

    length, D_outer, radius, thickness, d1_, d2_, b1_, b2_ = result.x
    rpm = 1500
    density = 2700 #7850
    E = 0.7e11 #2.0e11
    n_elem = 20

    d1 = int(np.clip(round(d1_), 0, n_elem))
    d2 = int(np.clip(round(d2_), 0, n_elem))
    b1 = int(np.clip(round(b1_), 0, n_elem))
    b2 = int(np.clip(round(b2_), 0, n_elem))

    beam = {'length': length, 'D_outer': D_outer, 'density': density, 'E': E, 'n_elem': n_elem}
    discs = [
        {'pos': d1, 'diameter': 2 * radius, 'thickness': thickness, 'density': density},
        {'pos': d2, 'diameter': 2 * radius, 'thickness': thickness, 'density': density}
    ]
    bearings = [{'pos': b1}, {'pos': b2}]
    Omega = rpm / 60 * 2 * np.pi

    return beam, discs, bearings, Omega, result


"""
def run_optimization(bounds):
    result = differential_evolution(objective, bounds, strategy='best1bin', maxiter=50, popsize=15, disp=True)

    # Unpack and round to 4 decimals
    length = round(result.x[0], 4)
    D_outer = round(result.x[1], 4)
    radius = round(result.x[2], 4)
    thickness = round(result.x[3], 4)
    d1_ = result.x[4]
    d2_ = result.x[5]
    b1_ = result.x[6]
    b2_ = result.x[7]
    rpm = round(result.x[8], 4)

    density = 7833.4
    E = 2.0684e11
    n_elem = 20

    # Clamp and round element positions
    d1 = int(np.clip(round(d1_), 0, n_elem))
    d2 = int(np.clip(round(d2_), 0, n_elem))
    b1 = int(np.clip(round(b1_), 0, n_elem))
    b2 = int(np.clip(round(b2_), 0, n_elem))

    beam = {
        'length': length,
        'D_outer': D_outer,
        'density': density,
        'E': E,
        'n_elem': n_elem
    }

    discs = [
        {'pos': d1, 'diameter': round(2 * radius, 4), 'thickness': thickness, 'density': density},
        {'pos': d2, 'diameter': round(2 * radius, 4), 'thickness': thickness, 'density': density}
    ]

    bearings = [{'pos': b1}, {'pos': b2}]
    Omega = round(rpm / 60 * 2 * np.pi, 4)

    return beam, discs, bearings, Omega, result
"""


# === Optional Test Run (only when executed directly) ===
"""
if __name__ == "__main__":
    default_bounds = [
        (0.9, 1.2), (0.005, 0.02), (0.06, 0.12), (0.005, 0.02),
        (5, 8), (14, 17), (5, 8), (14, 17), (100, 2000)
    ]
    beam, discs, bearings, Omega, result = run_optimization(default_bounds)
"""