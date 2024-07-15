import numpy as np
import matplotlib.pyplot as plt

class GenerateForceProfile:
    def __init__(self):
        # Initialize a 3D plot
        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')

    def t_func(self, tt):
        # Generate a time array with a specified number of points
        Np_Factor = 40
        return np.linspace(tt[0], tt[1], round(np.sum(np.abs(tt)) * Np_Factor / 100) * 100)

    def add_noise_to_normal_vectors(self, N, noise_level=0.01):
        # Add noise to normal vectors
        noise = noise_level * (np.random.rand(N.shape[0], N.shape[1]) - 0.5)
        return N + noise

    def get_magnitude(self, func, a, b, points):
        # Get magnitudes of a function over a specified range and number of points
        mag_values = np.linspace(a, b, points)
        return func(mag_values)

    def plot_trajectory_and_curvature(self, x, y, z, N, kappa, color="b"):
        # Plot the trajectory and curvature direction vectors
        scale_factor = 0.1
        magnitudes = np.linalg.norm(N, axis=1)
        for i in range(len(x)):
            end_point = np.array([x[i], y[i], z[i]])
            curvature_direction = scale_factor * kappa[i] * N[i]
            self.ax.quiver(end_point[0], end_point[1], end_point[2], curvature_direction[0], curvature_direction[1], curvature_direction[2], color=color, length=magnitudes[i], normalize=True)

    def get_refined_forces(self, original_force1, original_force2, k, alpha=-1.0, beta=-1.0):
        # Compute refined forces based on original forces, scaling factor, and coefficients
        return (original_force1/k)*alpha + (original_force2/k)*beta

    def generate_force_profile(self, example_id=1):
        # Generate a force profile based on the selected example ID
        if example_id == 0 or example_id is None:
            example_id = 1

        kappa_min_max = [0, 1.5]
        tau_min_max = [-1, 1]

        if example_id == 1:
            t = self.t_func([0, 2 * np.pi])
            r = 1
            x = r * np.sin(t)
            y = r * np.cos(t)
            z = r * np.cos(2 * t)

        elif example_id == 2:
            t = self.t_func([0, 2 * np.pi])
            r = 1
            x = t
            y = r * np.cos(t)
            z = r * np.cos(2 * t)

        elif example_id == 3:
            t = self.t_func([0, 2 * np.pi])
            r = 1
            x = t
            y = t
            z = r * np.cos(2 * t)

        # Calculate derivatives and curvatures
        dr = np.column_stack((np.gradient(x), np.gradient(y), np.gradient(z)))
        ds = np.linalg.norm(dr, axis=1)
        T = dr / ds[:, np.newaxis]
        dT = np.gradient(T, axis=0)
        dTds = dT / ds[:, np.newaxis]
        kappa = np.linalg.norm(dTds, axis=1)
        N = dTds / kappa[:, np.newaxis]
        return x, y, z, kappa, N

    def get_refined_trajectory(self, example_id=1, noise_level=0.21, k=15):

        # Generate the force profile for the given example ID
        x, y, z, kappa, N = self.generate_force_profile(example_id)

        # Get the magnitude of the force profile
        # force_mag = sin(t), where t is in the range [-0.2, 0.2] with the number of points equal to len(x)
        force_mag = self.get_magnitude(np.sin, -0.2, 0.2, len(x))

        # Compute the force profile as N.T * force_mag (element-wise multiplication)
        # \mathbf{F} = \mathbf{N}^T \cdot \text{force\_mag}^T
        force_profile = (N.T * force_mag).T

        # Add noise to the normal vectors
        # \mathbf{F}_{\text{noise}} = \mathbf{F} + \text{noise}
        force_profile_with_noise = self.add_noise_to_normal_vectors(force_profile, noise_level=noise_level)

        # Compute the resultant force as the sum of the original force profile and the noise
        # \mathbf{F}_{\text{resultant}} = \mathbf{F} + \mathbf{F}_{\text{noise}}
        resultant_force = force_profile + force_profile_with_noise

        # Calculate the deformation by dividing the resultant force by the stiffness constant k
        # \text{deformation} = \frac{\mathbf{F}_{\text{resultant}}}{k}
        deformation = resultant_force / k

        # Compute the deformed trajectory by adding the deformation to the original trajectory
        # \text{deformed\_traj} = \left[ x + \text{deformation}_x, y + \text{deformation}_y, z + \text{deformation}_z \right]^T
        deformed_traj = np.vstack((x + deformation[:, 0], y + deformation[:, 1], z + deformation[:, 2])).T

        # Refine the forces using the get_refined_forces function
        # \mathbf{F}_{\text{refined}} = \alpha \frac{\mathbf{F}}{k} + \beta \frac{\mathbf{F}_{\text{noise}}}{k}
        refined_forces = self.get_refined_forces(force_profile, force_profile_with_noise, k, alpha=-1.0, beta=-1.0)

        # Compute the refined trajectory by adding the refined forces to the original trajectory
        # \text{refined\_traj} = \left[ x + \mathbf{F}_{\text{refined}, x}, y + \mathbf{F}_{\text{refined}, y}, z + \mathbf{F}_{\text{refined}, z} \right]^T
        refined_traj = np.vstack((x + refined_forces[:, 0], y + refined_forces[:, 1], z + refined_forces[:, 2])).T


        # Plot initial, expected, and refined trajectories
        self.ax.plot(x, y, z, label='initial trajectory', color="k")
        self.plot_trajectory_and_curvature(x, y, z, force_profile, kappa, color=(1, 0.0, 0.0, 0.1))
        self.plot_trajectory_and_curvature(x, y, z, force_profile_with_noise, kappa, color=(0.0, 1.0, 0.0, 0.1))
        self.plot_trajectory_and_curvature(x, y, z, resultant_force, kappa, color=(0.0, 0.0, 1.0, 0.1))
        self.ax.plot(deformed_traj[:, 0], deformed_traj[:, 1], deformed_traj[:, 2], label='expected trajectory', color="y", linewidth=3.0)
        self.ax.plot(refined_traj[:, 0], refined_traj[:, 1], refined_traj[:, 2], label='refined trajectory', color="magenta", linewidth=3.0)

        self.ax.set_title(f'With Noise (level={noise_level})')
        self.ax.set_title('Trajectory and Curvature Direction')
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.legend()
        view = []
        self.ax.view_init(*view)

        plt.tight_layout()
        plt.show()


# Create an instance of the class and generate a refined trajectory
gen_force_profile = GenerateForceProfile()
gen_force_profile.get_refined_trajectory(example_id=3, noise_level=0.21, k=5)