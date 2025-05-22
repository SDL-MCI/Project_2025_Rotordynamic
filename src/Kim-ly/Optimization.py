# -*- coding: utf-8 -*-
"""
Created on Thu May 22 18:19:51 2025

@author: kimly
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh, eig
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.optimize import differential_evolution

# Bounds for optimization variables: [L, D_outer, D_inner, disc1_pos, disc2_pos, disc3_pos, disc_radius, disc_thick, bear1_pos, bear2_pos]

bounds = [
    (1500, 3000),      # Beam length [mm]
    (50, 200),         # Outer diameter [mm]
    (10, 100),         # Inner diameter [mm]
    (100, 2500),       # Disc 1 pos [mm]
    (100, 2500),       # Disc 2 pos [mm]
    (100, 2500),       # Disc 3 pos [mm]
    (30, 80),          # Disc radius [mm]
    (10, 100),         # Disc thickness [mm]
    (0, 2400),         # Bearing 1 pos [mm]
    (100, 2400)        # Bearing 2 pos [mm]
]

def evaluate_design(x):
    L, Douter, Dinner, disc1_pos, disc2_pos, disc3_pos, disc_diam, disc_thick, bear1_pos, bear2_pos = x
    Omega = 2000 / 60 * 2 * np.pi  # rad/s
    target_operating_freq = 2000 / 60  # ≈ 33.3 Hz
    safe_margin = 40  # Don't want any mode near this

    try:
        freqs = run_simulation(L, Douter, Dinner, disc1_pos, disc2_pos, disc3_pos,
                               disc_diam, disc_thick, bear1_pos, bear2_pos, Omega)
        f1, f2, f3 = freqs[:3]

        # Penalties:
        penalty = 0

        # 1. Penalize if modes are too close to 33.3 Hz (operating speed)
        for f in [f1, f2, f3]:
            if abs(f - target_operating_freq) < 10:
                penalty += 1e5  # Strong penalty

        # 2. Penalize if any mode exceeds 300 Hz or below safe margin
        for f in [f1, f2, f3]:
            if f < safe_margin:
                penalty += (safe_margin - f)**2 * 1e3
            elif f > 300:
                penalty += (f - 300)**2 * 1e3

        # 3. Encourage spread (mode separation)
        spacing_penalty = -((f2 - f1)**2 + (f3 - f2)**2)

        return penalty + spacing_penalty

    except Exception as e:
        print("Simulation error:", e)
        return 1e6  # Infeasible design


def run_simulation(L, Douter, Dinner, disc1_pos, disc2_pos, disc3_pos, disc_diam, disc_thick, bear1_pos, bear2_pos, Omega):
    
    # === USER INPUT FOR BEAM AND MATERIAL PROPERTIES ===
    rho = 7.8334e-6 # float(input("Density [kg/mm^3]: "))
    E = 2.0684e8 # float(input("Young's modulus [MPa]: "))

    I = (np.pi / 64) * (Douter**4 - Dinner**4)  # Moment of inertia for hollow circular cross-section [m^4]
    A = (np.pi / 4) * (Douter**2 - Dinner**2)   # Cross-sectional area for hollow cylinder [m^2]


    # === FEM discretization ===
    n_elem = 24
    n_nodes = n_elem + 1
    dof_per_node = 4                            # 4 DOF: u_y, theta_z, u_z, theta_y
    total_dof = dof_per_node * n_nodes
    dx = L / n_elem                             # Element length

    # === Element matrices (8x8) ===
    def beam_element_matrices_3D(E, I, rho, A, L):
        # 4x4 stiffness matrix for bending
        K_local = (E * I / L**3) * np.array([
            [12, 6*L, -12, 6*L],
            [6*L, 4*L**2, -6*L, 2*L**2],
            [-12, -6*L, 12, -6*L],
            [6*L, 2*L**2, -6*L, 4*L**2]
        ])
        
        # 4x4 consistent mass matrix
        M_local = (rho * A * L / 420) * np.array([
            [156, 22*L, 54, -13*L],
            [22*L, 4*L**2, 13*L, -3*L**2],
            [54, 13*L, 156, -22*L],
            [-13*L, -3*L**2, -22*L, 4*L**2]
        ])
        
        # 8x8 Element matrices in 3D
        K_e = np.zeros((8, 8))
        M_e = np.zeros((8, 8))
        
        # Fill 8x8 matrices (see notes)
        K_e[np.ix_([0,1,4,5],[0,1,4,5])] = K_local
        M_e[np.ix_([0,1,4,5],[0,1,4,5])] = M_local
        
        K_e[np.ix_([2,3,6,7],[2,3,6,7])] = K_local
        M_e[np.ix_([2,3,6,7],[2,3,6,7])] = M_local
        
        
        return K_e, M_e

    # === Initialize global matrices ===
    K_global = np.zeros((total_dof, total_dof))
    M_global = np.zeros((total_dof, total_dof))
    G_global = np.zeros((total_dof, total_dof))


    # === Assembly ===
    for e in range(n_elem):
        k_e, m_e = beam_element_matrices_3D(E, I, rho, A, dx)
        dof_map = [
            4*e, 4*e+1, 4*e+2, 4*e+3,
            4*e+4, 4*e+5, 4*e+6, 4*e+7
        ]
        
        for i in range(8):
            for j in range(8):
                K_global[dof_map[i], dof_map[j]] += k_e[i, j]
                M_global[dof_map[i], dof_map[j]] += m_e[i, j]
                              

    #=============================================================================================================================
    # === USER INPUT FOR 2 DISCS ===
    #print("\n=== Define 2 Rigid Discs ===")
    disc_density = 7833.4 # float(input("Disc 1 density [kg/m^3]: "))   # [kg/m³] (same as beam material)


    # === Add Discs ===
    # Store all disc properties in a list of dictionaries
    discs = [
        {'pos': disc1_pos, 'diameter': disc_diam, 'thickness': disc_thick, 'density': disc_density},
        {'pos': disc2_pos, 'diameter': disc_diam, 'thickness': disc_thick, 'density': disc_density},
        {'pos': disc3_pos, 'diameter': disc_diam, 'thickness': disc_thick, 'density': disc_density}
    ]

    # === ADD DISCS TO GLOBAL MASS MATRIX ===
    for disc in discs:
        r = disc['diameter'] / 2
        m = disc['density'] * np.pi * r**2 * disc['thickness']  # Mass of the disc
        J = (1/4) * m * r**2# + (1/12) * m * disc['thickness']**2  # Rotational inertia about diameter axis (appropriate for lateral bending)
        I = 0.5 * m * r**2 # Polar moment of inertia for the disc (used in gyroscopic effects)

        # Map position to nearest node
        node = int(np.clip(round(disc['pos'] / dx), 0, n_nodes - 1))
        # node = np.clip(disc['pos'], 0, n_nodes - 1)  # Ensure within bounds

        dof_uy = 4 * node   # v (Y translation)
        dof_theta_z = dof_uy + 1  # θz (rotation about Z)
        dof_uz= dof_uy + 2   # w (Z translation)
        dof_theta_y = dof_uy + 3 # θy (rotation about Y)

        # Add disc mass to translational DOFs
        M_global[dof_uy, dof_uy] += m
        M_global[dof_uz, dof_uz] += m
        
        # Add disc rotational inertia to rotational DOFs
        M_global[dof_theta_y, dof_theta_y] += J
        M_global[dof_theta_z, dof_theta_z] += J
        
        # Add gyroscopic matrix contributions
        G_local = np.array([
            [0,      0,      0, -I * Omega],
            [0,      0,  I * Omega, 0],
            [0, -I * Omega, 0,     0],
            [I * Omega, 0, 0,      0]
        ])

        # Map local [u_y, theta_z, u_z, theta_y] into global
        dof_indices = [dof_uy, dof_theta_z, dof_uz, dof_theta_y]

        for i in range(4):
            for j in range(4):
                G_global[dof_indices[i], dof_indices[j]] += G_local[i, j]


    # === USER INPUT FOR FLEXIBLE BEARINGS ===
     
    #print("\n=== Define 2 Bearings ===")

    # === Add Bearings ===
    # Store bearing properties
    
    bear1_node = int(np.clip(round(bear1_pos / dx), 0, n_nodes - 1))
    bear2_node = int(np.clip(round(bear2_pos / dx), 0, n_nodes - 1))
    
    bearings = [
        {'node': bear1_node},
        {'node': bear2_node}
    ]
    # bearings = [
    #     {'pos': bear1_pos},
    #     {'pos': bear2_pos}
    # ]
        
    #=============================================================================================================================  
       
    # === Boundary conditions (simply supported at both ends in Y and Z) ===
    # Fix displacements u_y, u_z at 2 bearings
    constrained_dofs = [
        4 * bear1_node, 4 * bear1_node + 2,   # u_y, u_z at bearing 1
        4 * bear2_node, 4 * bear2_node + 2    # u_y, u_z at bearing 2
    ]

    free_dofs = np.setdiff1d(np.arange(total_dof), constrained_dofs)

    # === Reduce matrices ===
    K_reduced = K_global[np.ix_(free_dofs, free_dofs)]
    M_reduced = M_global[np.ix_(free_dofs, free_dofs)]
    G_reduced = G_global[np.ix_(free_dofs, free_dofs)]
    I = np.eye(M_reduced.shape[0])                  # Identity matrix

    # Build A and B matrices
    A_mat = np.block([
        [G_reduced, M_reduced],
        [I, np.zeros_like(M_reduced)]
    ])

    B_mat = np.block([
        [K_reduced, np.zeros_like(M_reduced)],
        [np.zeros_like(M_reduced), -I]
    ])

    # Solve generalized eigenvalue problem: A x_dot + B x = 0
    eigvals, eigvecs = eig(-B_mat, A_mat)  # Notice negative sign to match equation

    # Extract natural frequencies from eigenvalues
    omega = np.abs(np.imag(eigvals))  # rad/s
    numerical_freqs = omega / (2 * np.pi)       # Hz

    # Ordering natural frequencies from lowest to highest
    sorted_indices = np.argsort(numerical_freqs)
    eigvecs = eigvecs[:, sorted_indices]
    numerical_freqs = numerical_freqs[sorted_indices]
        
    return numerical_freqs

result = differential_evolution(evaluate_design, bounds, maxiter=30, popsize=15)
print("Best parameters:", result.x)
print("Minimum objective value (Hz):", result.fun)

# === Re-run simulation for best design to get final frequencies ===
best_freqs = run_simulation(*result.x, Omega=2000/60 * 2 * np.pi)

print("\nFinal Optimized Natural Frequencies (Hz):")
for i, f in enumerate(best_freqs[:12]):
    print(f"w_{i+1}: {f:.2f} Hz")


"""
    # ====================================================== Print natural frequencies ======================================================
    print("Numerical Natural frequencies (Hz):")
    for i, f in enumerate(numerical_freqs[:12]):
        print(f"w_{i+1}: {f:.2f} Hz")

    # === Plot mode shapes (u_y and u_z) ===
    # ====================================================== PLot 3D ======================================================

    # Define number of modeshapes plotted
    n_modes = min(6, len(numerical_freqs))
    x = np.linspace(0, L, n_nodes)

    # creating a 3D Disc
    def create_disc(x_center, y_center, z_center, radius, num_points=30):
        theta = np.linspace(0, 2 * np.pi, num_points)
        y = y_center + radius * np.cos(theta)
        z = z_center + radius * np.sin(theta)
        x = np.full_like(y, x_center)
        return list(zip(x, y, z))

    # 3D-Plot for every modeshape
    for i in range(n_modes):
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')

        full_mode = np.zeros(total_dof)
        eigvec_displacement = eigvecs[M_reduced.shape[0]:, i]
        full_mode[free_dofs] = np.abs(eigvec_displacement)

        v = full_mode[0::4]
        w = full_mode[2::4]

        y_def = v
        z_def = w

        # Axis not deformed
        ax.plot(x, np.zeros_like(x), np.zeros_like(x), 'k--', linewidth=1, label='Axis')

        # Axis defomrmed
        ax.plot(x, y_def, z_def, 'y.-', label=f'Mode {i+1}')

        # Discs
        for disc in discs:
            node = int(disc['pos'])
            disc_points = create_disc(x[node], y_def[node], z_def[node], radius=0.05)
            disc_poly = Poly3DCollection([disc_points], color='skyblue', alpha=0.5)
            ax.add_collection3d(disc_poly)

        # Bearings

        for b in bearings:
            node = int(b['pos'])
            ax.scatter(x[node], y_def[node], z_def[node], c='r', s=60, label='Bearing' if b == bearings[0] else "")


        # Nodes
        ax.scatter(x, y_def, z_def, c='yellow', edgecolors='k', s=30, label='Node')
        for idx in range(n_nodes):
            ax.text(x[idx], y_def[idx], z_def[idx], str(idx), fontsize=6)

        ax.set_xlim(0, L)
        ax.set_ylim(-0.2, 0.2)
        ax.set_zlim(-0.2, 0.2)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f"3D Mode Shape {i+1} ({numerical_freqs[i]:.2f} Hz)")
        ax.view_init(elev=20, azim=135)
        ax.legend()

        plt.tight_layout()
        plt.show()

    

    # ====================================================== PLot 2D ======================================================
    x = np.linspace(0, L, n_nodes)
    n_modes = min(6, len(numerical_freqs))

    fig = plt.figure(figsize=(10, 2.5 * n_modes)) 
    for i in range(n_modes):
        full_mode = np.zeros(total_dof)
        eigvec_displacement = eigvecs[M_reduced.shape[0]:, i]
        full_mode[free_dofs] = np.abs(eigvec_displacement)

        v = full_mode[0::4]
        w = full_mode[2::4]

        ax = fig.add_subplot(n_modes, 1, i+1)
        plt.plot(x, v, '-o', label='u_y')
        plt.plot(x, w, '--', label='u_z')

        # === Mark disc positions (now using node indices directly) ===
        for disc in discs:
            disc_node = int(disc['pos'])
            disc_x = x[disc_node]
            plt.axvline(x=disc_x, color='k', linestyle='--', label='Disc' if i == 0 else "")
            plt.plot(disc_x, 0, 'ko', markerfacecolor='blue')

        # === Mark bearing positions (also node indices) ===
        for b in bearings:
            bear_node = int(b['pos'])
            bear_x = x[bear_node]
            plt.axvline(x=bear_x, color='g', linestyle='-.', label='Bearing' if i == 0 else "")
            plt.plot(bear_x, 0, 'go', markerfacecolor='orange')

        plt.title(f"Mode {i+1}: {numerical_freqs[i]:.2f} Hz")
        plt.xlabel("Beam length (m)")
        plt.ylabel("Displacement")
        plt.legend(loc='upper right')
        plt.grid(True)

    plt.tight_layout()
    plt.show()
    """
