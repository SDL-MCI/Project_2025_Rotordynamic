import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eig

class RotorSystem:
    def __init__(self, beam, discs, bearings, Omega):
        self.L = beam['length']
        self.Douter = beam['D_outer']
        self.rho = beam['density']
        self.E = beam['E']
        self.n_elem = beam['n_elem']
        self.Omega = Omega

        self.discs = discs
        self.bearings = bearings

        self.I = (np.pi / 64) * (self.Douter**4)
        self.A = (np.pi / 4) * (self.Douter**2)
        self.n_nodes = self.n_elem + 1
        self.dof_per_node = 4
        self.total_dof = self.n_nodes * self.dof_per_node
        self.dx = self.L / self.n_elem

        self.M_global = np.zeros((self.total_dof, self.total_dof))
        self.K_global = np.zeros((self.total_dof, self.total_dof))
        self.G_global = np.zeros((self.total_dof, self.total_dof))

    def beam_element_matrices_3D(self):
        l = self.dx
        E, I = self.E, self.I
        rho, A = self.rho, self.A
        Douter= self.Douter

        Theta_p = 0.5 * rho * A * ((Douter / 2)**2)
        Theta_d = 0.25 * rho * A * ((Douter / 2)**2)

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

        M = Mr + Mt

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

    def assemble_global_matrices(self):
        for e in range(self.n_elem):
            K_e, M_e, G_e = self.beam_element_matrices_3D()
            dof_map = [
                4*e+4, 4*e+5, 4*e+6, 4*e+7,
                4*e+0, 4*e+1, 4*e+2, 4*e+3
            ]

            for i in range(8):
                for j in range(8):
                    self.K_global[dof_map[i], dof_map[j]] += K_e[i, j]
                    self.M_global[dof_map[i], dof_map[j]] += M_e[i, j]
                    self.G_global[dof_map[i], dof_map[j]] += G_e[i, j]

        self.add_discs()

    def add_discs(self):
            for disc in self.discs:
                r = disc['diameter'] / 2
                m = disc['density'] * np.pi * r**2 * disc['thickness']
                Theta_d = (1/4) * m * r**2
                Theta_p = 0.5 * m * r**2

                # Local mass matrix for the disc: diag([m, Theta_p, m, Theta_p])
                M_local = np.diag([m, Theta_p, m, Theta_p])

                # Local gyroscopic matrix (only 2,4 and 4,2 elements nonzero)
                G_local = np.zeros((4, 4))
                G_local[1, 3] = -Theta_d
                G_local[3, 1] = Theta_d

                # Map position to nearest node
                node = int(np.clip(disc['pos'], 0, self.n_nodes))  # Ensure within bounds
                # node = int(np.clip(disc['pos'], 0, self.n_nodes - 1))


                dof_uy = 4 * node      # v (Y translation)
                dof_theta_z = dof_uy + 1  # θz (rotation about Z)
                dof_uz = dof_uy + 2    # w (Z translation)
                dof_theta_y = dof_uy + 3  # θy (rotation about Y)

                # Assemble into global matrices
                dofs = [dof_uy, dof_theta_z, dof_uz, dof_theta_y]

                for i in range(4):
                    for j in range(4):
                        self.M_global[dofs[i], dofs[j]] += M_local[i, j]
                        self.G_global[dofs[i], dofs[j]] += G_local[i, j]


    def apply_boundary_conditions(self):

        self.constrained_dofs = [
            4 * self.bearings[0]['pos'], 4 * self.bearings[0]['pos'] + 2,   # u_y, u_z at bearing 1
            4 * self.bearings[1]['pos'], 4 * self.bearings[1]['pos'] + 2    # u_y, u_z at bearing 2
        ]

        all_dofs = np.arange(self.total_dof)
        self.free_dofs = np.setdiff1d(all_dofs, self.constrained_dofs)

        for dof in self.constrained_dofs:
            self.M_global[dof, :] = 0.0
            self.M_global[:, dof] = 0.0
            self.M_global[dof, dof] = 1.0  # Set diagonal to 1 for constrained DOFs

            self.K_global[dof, :] = 0.0
            self.K_global[:, dof] = 0.0
            self.K_global[dof, dof] = 1.0  # Set diagonal to 1 for constrained DOFs

            self.G_global[dof, :] = 0.0
            self.G_global[:, dof] = 0.0
            self.G_global[dof, dof] = 0.0  # Set diagonal to 0 for constrained DOFs

        self.M_reduced = self.M_global
        self.K_reduced = self.K_global
        self.G_reduced = self.G_global

    def solve_eigenproblem(self):
        I = np.eye(len(self.M_reduced[0]))

        A = np.block([
            [self.Omega * self.G_reduced, self.M_reduced],
            [I, np.zeros_like(self.M_reduced)]
        ])

        B = np.block([
            [self.K_reduced, np.zeros_like(self.M_reduced)],
            [np.zeros_like(self.M_reduced), -I]
        ])

        eigvals, eigvecs = eig(B, -A)
        omega = np.abs(np.imag(eigvals))
        self.natural_frequencies = omega / (2 * np.pi)

        idx = np.argsort(self.natural_frequencies)
        eigvecs = eigvecs[:, idx]
        self.natural_frequencies = self.natural_frequencies[idx]
        
        threshold = 0.2 * 2.0 * np.pi   # Threshold to filter out BCs
        keep_mask = self.natural_frequencies >= threshold    # Boolean of what to keep
        
        # Get the indices that are kept & filtered_out
        kept_idx = np.where(keep_mask)[0]
        filtered_out_idx = np.where(~keep_mask)[0]

        # Filter natural_frequencies and eigvecs 
        self.natural_frequencies = self.natural_frequencies[kept_idx]
        self.mode_shapes = eigvecs[self.M_reduced.shape[0]:, kept_idx]
        self.free_dofs = np.setdiff1d(np.arange(self.total_dof), filtered_out_idx)
        
        
        #self.natural_frequencies = [f for f in self.natural_frequencies if not (f < 0.2*2.0*np.pi)]  # Filter out BCs

        

    def get_frequencies(self):
        return self.natural_frequencies

    def get_mode_shapes(self):
        return self.mode_shapes, self.free_dofs


# === Sensitivity Analysis ===

def clone_rotor(rotor):
    beam = {
        'length': rotor.L,
        'D_outer': rotor.Douter,
        'density': rotor.rho,
        'E': rotor.E,
        'n_elem': rotor.n_elem
    }
    discs = [disc.copy() for disc in rotor.discs]
    bearings = [b.copy() for b in rotor.bearings]
    return RotorSystem(beam, discs, bearings, rotor.Omega)


def sensitivity_analysis(rotor_template, param_defs, delta=0.05, n_modes=6):
    base_freqs = rotor_template.get_frequencies()[:n_modes]
    sensitivities = []

    for name, get_func, set_func in param_defs:
        original = get_func(rotor_template)
        perturbed_up = original * (1 + delta)
        perturbed_down = original * (1 - delta)

        rotor_up = clone_rotor(rotor_template)
        set_func(rotor_up, perturbed_up)
        rotor_up.assemble_global_matrices()
        rotor_up.apply_boundary_conditions()
        rotor_up.solve_eigenproblem()
        freqs_up = rotor_up.get_frequencies()[:n_modes]

        rotor_down = clone_rotor(rotor_template)
        set_func(rotor_down, perturbed_down)
        rotor_down.assemble_global_matrices()
        rotor_down.apply_boundary_conditions()
        rotor_down.solve_eigenproblem()
        freqs_down = rotor_down.get_frequencies()[:n_modes]

        sens = (np.array(freqs_up) - np.array(freqs_down)) / (2 * delta * original)
        sensitivities.append((name, sens))

    return sensitivities


def plot_sensitivity_bars(sensitivity_data):
    param_names = [name for name, _ in sensitivity_data]
    mode_count = len(sensitivity_data[0][1])
    x = np.arange(len(param_names))
    width = 0.1

    plt.figure(figsize=(14, 6))
    for i in range(mode_count):
        mode_vals = [s[1][i] for s in sensitivity_data]
        plt.bar(x + i * width, mode_vals, width, label=f'Mode {i+1}')

    plt.xticks(x + width * (mode_count - 1) / 2, param_names, rotation=45)
    plt.ylabel('Sensitivity (Hz / % change)')
    plt.title('Sensitivity of Natural Frequencies to Parameter Changes')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()