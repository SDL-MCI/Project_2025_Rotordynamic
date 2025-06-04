#!/usr/bin/env python
# coding: utf-8

# # MODIFIED PARAMETERS

# In[31]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh, eig
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# === USER INPUT FOR BEAM AND MATERIAL PROPERTIES ===
print("\n=== Define the beam and material properties ===")
L = 1.0 # float(input("Total beam length [m]: "))
Dinner = 0.0 # float(input("Inner diameter of beam [m]: "))
Douter  = 16e-3 # float(input("Outer diameter of beam [m]: "))
rho = 7.85e3 # float(input("Density [kg/m^3]: "))
E = 2.0e11 # float(input("Young's modulus [Pa]: "))
Omega = 1000/60*2*np.pi    #1000/60*2*np.pi # float(input("Spin speed [rad/s]: "))

I = (np.pi / 64) * (Douter**4 - Dinner**4)  # Moment of inertia for hollow circular cross-section [m^4]
A = (np.pi / 4) * (Douter**2 - Dinner**2)   # Cross-sectional area for hollow cylinder [m^2]


# === FEM discretization ===
n_elem = 24
n_nodes = n_elem + 1
dof_per_node = 4                            # 4 DOF: u_y, theta_z, u_z, theta_y
total_dof = dof_per_node * n_nodes
l = L / n_elem                             # Element length

def beam_element_matrices_3D(E, I, rho, A, l):

    # Mass moments of inertia
    # Polar mass moment of inertia for the beam
    Theta_p = 0.5*rho*A*((Douter/2)**2 - (Dinner/2)**2);
    
    # Diametral mass moment of inertia
    Theta_d = 0.25*rho*A*((Douter/2)**2 - (Dinner/2)**2);

    # Precompute terms for mass matrix Mt
    t2 = l**2
    t3 = l**3
    t4 = A * l * rho * (13.0 / 35.0)
    t5 = A * l * rho * (9.0 / 70.0)
    t6 = (A * rho * t3) / 105.0
    t7 = (A * rho * t3) / 140.0
    t8 = -t7
    t9 = A * rho * t2 * (11.0 / 210.0)
    t10 = A * rho * t2 * (13.0 / 420.0)
    t11 = -t9
    t12 = -t10

    Mt = np.array([
        [t4,t11,0,0,t5,t10,0,0],
        [t11,t6,0,0,t12,t8,0,0],
        [0,0,t4,t11,0,0,t5,t10],
        [0,0,t11,t6,0,0,t12,t8],
        [t5,t12,0,0,t4,t9,0,0],
        [t10,t8,0,0,t9,t6,0,0],
        [0,0,t5,t12,0,0,t4,t9],
        [0,0,t10,t8,0,0,t9,t6]
    ])

    # Precompute terms for rotary mass matrix Mr
    t2 = 1.0 / l
    t3 = Theta_p / 10.0
    t4 = Theta_p * l * (2.0 / 15.0)
    t5 = (Theta_p * l) / 30.0
    t6 = -t3
    t7 = -t5
    t8 = Theta_p * t2 * (6.0 / 5.0)
    t9 = -t8

    Mr = np.array([
        [t8,t6,0,0,t9,t6,0,0],
        [t6,t4,0,0,t3,t7,0,0],
        [0,0,t8,t6,0,0,t9,t6],
        [0,0,t6,t4,0,0,t3,t7],
        [t9,t3,0,0,t8,t3,0,0],
        [t6,t7,0,0,t3,t4,0,0],
        [0,0,t9,t3,0,0,t8,t3],
        [0,0,t6,t7,0,0,t3,t4]
    ])

    # Total mass matrix
    M = Mr + Mt

    # Gyroscopic matrix
    t2 = 1.0 / l
    t3 = Theta_d / 10.0
    t4 = -t3
    t5 = Theta_d * l * (2.0 / 15.0)
    t6 = (Theta_d * l) / 30.0
    t7 = -t5
    t8 = -t6
    t9 = Theta_d * t2 * (6.0 / 5.0)
    t10 = -t9

    G = np.array([
        [0,0,t9,t4,0,0,t10,t4],
        [0,0,t4,t5,0,0,t3,t8],
        [t10,t3,0,0,t9,t3,0,0],
        [t3,t7,0,0,t4,t6,0,0],
        [0,0,t10,t3,0,0,t9,t3],
        [0,0,t4,t8,0,0,t3,t5],
        [t9,t4,0,0,t10,t4,0,0],
        [t3,t6,0,0,t4,t7,0,0]
    ])

    # Stiffness matrix
    t2 = 1.0 / l
    t3 = t2**2
    t4 = t2**3
    t5 = 2 * E * I * t2
    t6 = 4 * E * I * t2
    t7 = 6 * E * I * t3
    t8 = -t7
    t9 = 12 * E * I * t4
    t10 = -t9

    K = np.array([
        [t9,t8,0,0,t10,t8,0,0],
        [t8,t6,0,0,t7,t5,0,0],
        [0,0,t9,t8,0,0,t10,t8],
        [0,0,t8,t6,0,0,t7,t5],
        [t10,t7,0,0,t9,t7,0,0],
        [t8,t5,0,0,t7,t6,0,0],
        [0,0,t10,t7,0,0,t9,t7],
        [0,0,t8,t5,0,0,t7,t6]
    ])

    return K, M, G


# === Initialize global matrices ===
K_global = np.zeros((total_dof, total_dof))
M_global = np.zeros((total_dof, total_dof))
G_global = np.zeros((total_dof, total_dof))


# === Assembly ===
for e in range(n_elem):
    k_e, m_e, g_e = beam_element_matrices_3D(E, I, rho, A, l)
    dof_map = [
        4*e+4, 4*e+5, 4*e+6, 4*e+7,
        4*e+0, 4*e+1, 4*e+2, 4*e+3
    ]

    for i in range(8):
        for j in range(8):
            K_global[dof_map[i], dof_map[j]] += k_e[i, j]
            M_global[dof_map[i], dof_map[j]] += m_e[i, j]
            G_global[dof_map[i], dof_map[j]] += g_e[i, j]
                          

#=============================================================================================================================
# === USER INPUT FOR 2 DISCS ===
print("\n=== Define 2 Rigid Discs ===")
disc1_pos = 6-1  # float(input("Disc 1 position [m]: "))   # [m] position along beam
disc1_diam = 0.2    # float(input("Disc 1 diameter [m]: "))   # [m] diameter of rigid disc
disc1_thick = 0.01  # float(input("Disc 1 thickness [m]: "))   # [m] axial thickness
disc1_density = 7850 # float(input("Disc 1 density [kg/m^3]: "))   # [kg/m³] (same as beam material)

disc2_pos = 14-1 # float(input("Disc 2 position [m]: "))
disc2_diam = 0.2      # float(input("Disc 2 diameter [m]: "))
disc2_thick = 0.01  # float(input("Disc 2 thickness [m]: "))
disc2_density = 7850 # float(input("Disc 2 density [kg/m^3]: "))

#disc3_pos = 16 # float(input("Disc 2 position [m]: "))
#disc3_diam = 0.1047 # float(input("Disc 2 diameter [m]: "))
#disc3_thick = 0.04474 # float(input("Disc 2 thickness [m]: "))
#disc3_density = 7833.4 # float(input("Disc 2 density [kg/m^3]: "))


# === Add Discs ===
# Store all disc properties in a list of dictionaries
#discs = [
#    {'pos': disc1_pos, 'diameter': disc1_diam, 'thickness': disc1_thick, 'density': disc1_density},
#    {'pos': disc2_pos, 'diameter': disc2_diam, 'thickness': disc2_thick, 'density': disc2_density},
#    {'pos': disc3_pos, 'diameter': disc3_diam, 'thickness': disc3_thick, 'density': disc3_density}
#]

discs = [
    {'pos': disc1_pos, 'diameter': disc1_diam, 'thickness': disc1_thick, 'density': disc1_density},
    {'pos': disc2_pos, 'diameter': disc2_diam, 'thickness': disc2_thick, 'density': disc2_density},
]

for disc in discs:
    r = disc['diameter'] / 2
    m = disc['density'] * np.pi * r**2 * disc['thickness']  # Mass of the disc
    Theta_d = (1/4) * m * r**2 #+ (1/12) * m * disc['thickness']**2  # Rotational inertia about diameter axis (appropriate for lateral bending)
    Theta_p =  0.5 * m * r**2 # Polar moment of inertia for the disc (used in gyroscopic effects)
    

    # Local mass matrix for the disc: diag([m, Theta_p, m, Theta_p])
    M_local = np.diag([m, Theta_p, m, Theta_p])

    # Local gyroscopic matrix (only 2,4 and 4,2 elements nonzero)
    G_local = np.zeros((4, 4))
    G_local[1, 3] = -Theta_d
    G_local[3, 1] = Theta_d

    # Map position to nearest node
    node = int(np.clip(disc['pos'], 0, n_nodes - 1))  # Ensure within bounds

    dof_uy = 4 * node      # v (Y translation)
    dof_theta_z = dof_uy + 1  # θz (rotation about Z)
    dof_uz = dof_uy + 2    # w (Z translation)
    dof_theta_y = dof_uy + 3  # θy (rotation about Y)

    # Assemble into global matrices
    dof_indices = [dof_uy, dof_theta_z, dof_uz, dof_theta_y]

    for i in range(4):
        for j in range(4):
            M_global[dof_indices[i], dof_indices[j]] += M_local[i, j]
            G_global[dof_indices[i], dof_indices[j]] += G_local[i, j]

# === USER INPUT FOR FLEXIBLE BEARINGS ===

print("\n=== Define 2 Bearings ===")
bear1_pos = 4-1 # float(input("Bearing 1 position [m]: "))   # [m] position along beam
bear2_pos = 22-1 # float(input("Bearing 2 position [m]: "))

# === Add Bearings ===
# Store bearing properties
bearings = [
    {'pos': bear1_pos},
    {'pos': bear2_pos}
]
    
#=============================================================================================================================  
   
# === Boundary conditions (simply supported at both ends in Y and Z) ===
# Fix displacements u_y, u_z at 2 bearings
constrained_dofs = [
    4 * bear1_pos, 4 * bear1_pos + 2,   # u_y, u_z at bearing 1
    4 * bear2_pos, 4 * bear2_pos + 2    # u_y, u_z at bearing 2
]

free_dofs = np.setdiff1d(np.arange(total_dof), constrained_dofs)
print(total_dof)

for dof in constrained_dofs:
    M_global[dof, :] = 0.0
    M_global[:, dof] = 0.0
    M_global[dof, dof] = 1.0  # Set diagonal to 1 for constrained DOFs

    K_global[dof, :] = 0.0
    K_global[:, dof] = 0.0
    K_global[dof, dof] = 1.0  # Set diagonal to 1 for constrained DOFs

    G_global[dof, :] = 0.0
    G_global[:, dof] = 0.0
    G_global[dof, dof] = 0.0  # Set diagonal to 0 for constrained DOFs

M_reduced = M_global
K_reduced = K_global
G_reduced = G_global

eigvals0, phi0 = eig(K_reduced, M_reduced)
fn0 = np.sqrt(np.real(eigvals0))/2.0/np.pi
fn0 = np.sort(fn0)
fn0 = [f for f in fn0 if not (f > 0.99/(2.0*np.pi) and f < 1.01/(2.0*np.pi))]  # Filter out BCs

print("Natural frequencies (Hz)")
print(fn0[0:10])

I = np.eye(M_reduced.shape[0])                  # Identity matrix

# Build A and B matrices
M_hat = np.block([
    [Omega * G_reduced, M_reduced],
    [I, np.zeros_like(M_reduced)]
])

K_hat = np.block([
    [K_reduced, np.zeros_like(M_reduced)],
    [np.zeros_like(M_reduced), -I]
])


# Solve generalized eigenvalue problem: A x_dot + B x = 0
eigvals, eigvecs = eig(K_hat, -M_hat)  # Notice negative sign to match equation

# Extract natural frequencies from eigenvalues
omega = np.abs(np.imag(eigvals))  # rad/s
numerical_freqs = omega / (2 * np.pi)       # Hz

# Ordering natural frequencies from lowest to highest
sorted_indices = np.argsort(numerical_freqs)
eigvecs = eigvecs[:, sorted_indices]
numerical_freqs = numerical_freqs[sorted_indices]
numerical_freqs = [f for f in numerical_freqs if not (f < 0.2*2.0*np.pi)]  # Filter out BCs

# ====================================================== Print natural frequencies ======================================================
print("Numerical Natural frequencies (Hz):")
for i, f in enumerate(numerical_freqs[:12]):
    print(f"w_{i+1}: {f:.2f} Hz")

# === Plot mode shapes (u_y and u_z) ===
# ====================================================== PLot 3D ======================================================

# ============================================================================================
# ============================================================================================
# ========TODO: Fix extraction of mode shapes now that the system is not reduced again========
# ============================================================================================
# ============================================================================================

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
    #eigvec_displacement = eigvecs[M_global.shape[0]:, i]
    #full_mode[total_dof] = np.abs(eigvec_displacement)

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

# import numpy as np

# E = 2.0684e11      # Pa
# rho = 7833.4       # kg/m^3
# L = 0.9502         # m
# r = 0.008          # m

# I = (np.pi / 4) * r**4           # m^4 (simplified for solid circle)
# A = np.pi * r**2                 # m^2

# def analytical_freq(n, E, I, rho, A, L):
#     beta_n = n * np.pi
#     return (beta_n**2 / (2 * np.pi * L**2)) * np.sqrt(E * I / (rho * A))

# for n in range(1, 5):
#     f = analytical_freq(n, E, I, rho, A, L)
#     print(f"Mode {n}: {f:.2f} Hz")
"""

