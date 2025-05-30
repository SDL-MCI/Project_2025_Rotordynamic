import numpy as np
from scipy.linalg import eig

class RotorSystem:
    def __init__(self, beam, discs, bearings, Omega=0):
        self.L = beam['length']
        self.Dinner = beam['D_inner']
        self.Douter = beam['D_outer']
        self.rho = beam['density']
        self.E = beam['E']
        self.n_elem = beam['n_elem']
        self.Omega = Omega

        self.discs = discs
        self.bearings = bearings

        self.I = (np.pi / 64) * (self.Douter**4 - self.Dinner**4)
        self.A = (np.pi / 4) * (self.Douter**2 - self.Dinner**2)
        self.n_nodes = self.n_elem + 1
        self.dof_per_node = 4
        self.total_dof = self.n_nodes * self.dof_per_node
        self.dx = self.L / self.n_elem

        self.M_global = np.zeros((self.total_dof, self.total_dof))
        self.K_global = np.zeros((self.total_dof, self.total_dof))
        self.G_global = np.zeros((self.total_dof, self.total_dof))

    def beam_element_matrices_3D(self):
        L = self.dx
        E, I = self.E, self.I
        rho, A = self.rho, self.A

        K_local = (E * I / L**3) * np.array([
            [12, 6*L, -12, 6*L],
            [6*L, 4*L**2, -6*L, 2*L**2],
            [-12, -6*L, 12, -6*L],
            [6*L, 2*L**2, -6*L, 4*L**2]
        ])

        M_local = (rho * A * L / 420) * np.array([
            [156, 22*L, 54, -13*L],
            [22*L, 4*L**2, 13*L, -3*L**2],
            [54, 13*L, 156, -22*L],
            [-13*L, -3*L**2, -22*L, 4*L**2]
        ])

        K_e = np.zeros((8, 8))
        M_e = np.zeros((8, 8))

        K_e[np.ix_([0,1,4,5],[0,1,4,5])] = K_local
        M_e[np.ix_([0,1,4,5],[0,1,4,5])] = M_local
        K_e[np.ix_([2,3,6,7],[2,3,6,7])] = K_local
        M_e[np.ix_([2,3,6,7],[2,3,6,7])] = M_local

        return K_e, M_e

    def assemble_global_matrices(self):
        for e in range(self.n_elem):
            K_e, M_e = self.beam_element_matrices_3D()
            dof_map = [4*e+i for i in range(8)]

            for i in range(8):
                for j in range(8):
                    self.K_global[dof_map[i], dof_map[j]] += K_e[i, j]
                    self.M_global[dof_map[i], dof_map[j]] += M_e[i, j]

        self.add_discs()

    def add_discs(self):
        for disc in self.discs:
            r = disc['diameter'] / 2
            m = disc['density'] * np.pi * r**2 * disc['thickness']
            J = (1/4) * m * r**2 + (1/12) * m * disc['thickness']**2
            I_gyro = 0.5 * m * r**2

            node = int(np.clip(disc['pos'], 0, self.n_nodes - 1))
            dofs = [4*node + i for i in range(4)]

            self.M_global[dofs[0], dofs[0]] += m
            self.M_global[dofs[2], dofs[2]] += m
            self.M_global[dofs[1], dofs[1]] += J
            self.M_global[dofs[3], dofs[3]] += J

            G_local = np.array([
                [0, 0, 0, -I_gyro * self.Omega],
                [0, 0, I_gyro * self.Omega, 0],
                [0, -I_gyro * self.Omega, 0, 0],
                [I_gyro * self.Omega, 0, 0, 0]
            ])

            for i in range(4):
                for j in range(4):
                    self.G_global[dofs[i], dofs[j]] += G_local[i, j]

    def apply_boundary_conditions(self):
        self.constrained_dofs = []
        for bearing in self.bearings:
            node = int(np.clip(bearing['pos'], 0, self.n_nodes - 1))
            self.constrained_dofs += [4*node, 4*node+2]  # uy and uz

        all_dofs = np.arange(self.total_dof)
        self.free_dofs = np.setdiff1d(all_dofs, self.constrained_dofs)

        self.K_red = self.K_global[np.ix_(self.free_dofs, self.free_dofs)]
        self.M_red = self.M_global[np.ix_(self.free_dofs, self.free_dofs)]
        self.G_red = self.G_global[np.ix_(self.free_dofs, self.free_dofs)]

    def solve_eigenproblem(self):
        I = np.eye(len(self.M_red))

        A = np.block([
            [self.G_red, self.M_red],
            [I, np.zeros_like(self.M_red)]
        ])

        B = np.block([
            [self.K_red, np.zeros_like(self.M_red)],
            [np.zeros_like(self.M_red), -I]
        ])

        eigvals, eigvecs = eig(-B, A)
        omega = np.abs(np.imag(eigvals))
        self.natural_frequencies = omega / (2 * np.pi)

        idx = np.argsort(self.natural_frequencies)
        self.natural_frequencies = self.natural_frequencies[idx]
        self.mode_shapes = eigvecs[self.M_red.shape[0]:, idx]


    def get_frequencies(self):
        return self.natural_frequencies

    def get_mode_shapes(self):
        return self.mode_shapes, self.free_dofs
