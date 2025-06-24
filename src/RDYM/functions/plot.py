import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from functions.rotorsystem import RotorSystem

class ModePlotter:
    def __init__(self, rotor_system):
        self.rotor = rotor_system
        self.L = rotor_system.L
        self.n_nodes = rotor_system.n_nodes
        self.total_dof = rotor_system.total_dof
        self.x = np.linspace(0, self.L, self.n_nodes)

    def plot_2D_modes(self, n_modes=6):
        freqs = self.rotor.get_frequencies()
        mode_shapes, free_dofs = self.rotor.get_mode_shapes()
        n_modes = min(n_modes, len(freqs))

        fig = plt.figure(figsize=(10, 2.5 * n_modes)) 
        for i in range(n_modes):
            full_mode = np.zeros(self.total_dof)
            full_mode = mode_shapes[:, i]
            #full_mode[free_dofs] = np.abs(eigvec_displacement[free_dofs])

            v = np.real(full_mode[0::4])
            w = np.real(full_mode[2::4])

            ax = fig.add_subplot(n_modes, 1, i+1)
            plt.plot(self.x, v, '-o', label='u_y')
            plt.plot(self.x, w, '--', label='u_z')

            for disc in self.rotor.discs:
                node = int(disc['pos'])
                disc_x = self.x[node]
                plt.axvline(x=disc_x, color='k', linestyle='--', label='Disc' if i == 0 else "")
                plt.plot(disc_x, 0, 'ko', markerfacecolor='blue')

            for b in self.rotor.bearings:
                node = int(b['pos'])
                x_b = self.x[node]
                plt.axvline(x=x_b, color='g', linestyle='-.', label='Bearing' if i == 0 else "")
                plt.plot(x_b, 0, 'go', markerfacecolor='orange')

            plt.title(f"Mode {i+1}: {freqs[i]:.2f} Hz")
            plt.xlabel("Beam length (m)")
            plt.ylabel("Displacement")
            plt.legend(loc='upper right')
            plt.grid(True)

        plt.tight_layout()
        plt.show()

    def plot_3D_modes(self, n_modes=6):
        from mpl_toolkits.mplot3d import Axes3D  # only used in this function

        freqs = self.rotor.get_frequencies()
        mode_shapes, free_dofs = self.rotor.get_mode_shapes()
        n_modes = min(n_modes, len(freqs))
        x = self.x

        for i in range(n_modes):
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection='3d')

            full_mode = np.zeros(self.total_dof)
            full_mode = mode_shapes[:, i]
            #eigvec_displacement = mode_shapes[self.total_dof:, i]
            #full_mode[free_dofs] = np.abs(eigvec_displacement[free_dofs])

            v = np.real(full_mode[0::4])
            w = np.real(full_mode[2::4])
            y_def = v
            z_def = w

            ax.plot(x, np.zeros_like(x), np.zeros_like(x), 'k--', linewidth=1, label='Axis')
            ax.plot(x, y_def, z_def, 'y.-', label=f'Mode {i+1}')

            for disc in self.rotor.discs:
                node = int(disc['pos'])
                disc_points = self.create_disc(x[node], y_def[node], z_def[node], radius=0.05)
                disc_poly = Poly3DCollection([disc_points], color='skyblue', alpha=0.5)
                ax.add_collection3d(disc_poly)

            for b in self.rotor.bearings:
                node = int(b['pos'])
                ax.scatter(x[node], y_def[node], z_def[node], c='r', s=60, label='Bearing' if b == self.rotor.bearings[0] else "")

            ax.scatter(x, y_def, z_def, c='yellow', edgecolors='k', s=30, label='Node')
            for idx in range(self.n_nodes):
                ax.text(x[idx], y_def[idx], z_def[idx], str(idx), fontsize=6)

            ax.set_xlim(0, self.L)
            ax.set_ylim(-0.2, 0.2)
            ax.set_zlim(-0.2, 0.2)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title(f"3D Mode Shape {i+1} ({freqs[i]:.2f} Hz)")
            ax.view_init(elev=20, azim=135)
            ax.legend()
            plt.tight_layout()
            plt.show()

    def create_disc(self, x_center, y_center, z_center, radius, num_points=30):
        theta = np.linspace(0, 2 * np.pi, num_points)
        y = y_center + radius * np.cos(theta)
        z = z_center + radius * np.sin(theta)
        x = np.full_like(y, x_center)
        return list(zip(x, y, z))
    
# === Campbell diagram ===

def compute_campbell_diagram(rotor_template, rpm_range, n_modes=12):
    campbell_data = []

    for rpm in rpm_range:
        # Rebuild beam dictionary
        beam_dict = {
            'length': rotor_template.L,
            'D_outer': rotor_template.Douter,
            'density': rotor_template.rho,
            'E': rotor_template.E,
            'n_elem': rotor_template.n_elem
        }

        # Copy discs and bearings
        discs_copy = [disc.copy() for disc in rotor_template.discs]
        bearings_copy = [bear.copy() for bear in rotor_template.bearings]

        Omega = rpm / 60 * 2 * np.pi
        rotor = RotorSystem(beam_dict, discs_copy, bearings_copy, Omega)

        rotor.assemble_global_matrices()
        rotor.apply_boundary_conditions()
        try:
            rotor.solve_eigenproblem()
            freqs = rotor.get_frequencies()
            campbell_data.append(freqs[:n_modes])
        except Exception as e:
            print(f"⚠️ Failed at {rpm} RPM: {e}")
            campbell_data.append([np.nan] * n_modes)

    return np.array(campbell_data)


def plot_campbell_diagram(rpm_range, campbell_data):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(14, 6))
    for i in range(campbell_data.shape[1]):
        plt.plot(rpm_range, campbell_data[:, i], label=f'f_{i+1}')

    plt.plot(rpm_range, rpm_range / 60, 'k--', label='1x speed')
    plt.plot(rpm_range, 2 * rpm_range / 60, 'k:', label='2x speed')
    plt.xlabel('Rotational Speed [RPM]')
    plt.ylabel('Natural Frequencies [Hz]')
    plt.title('Campbell Diagram')
    plt.grid(True)
    plt.legend(loc='upper right', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
    plt.tight_layout()
    plt.show()