import numpy as np
import matplotlib.pyplot as plt

class GenerateForceProfile:
    def __init__(self):
        pass

    def t_func(self, tt):
        Np_Factor = 40
        return np.linspace(tt[0], tt[1], round(np.sum(np.abs(tt)) * Np_Factor / 100) * 100)

    def add_noise_to_normal_vectors(self, N, noise_level=0.01):
        noise = noise_level * (np.random.rand(N.shape[0], N.shape[1]) - 0.5)
        return N + noise
    
    def get_magnitude(self, func, a, b, points):
        mag_values = np.linspace(a, b, points)
        return func(mag_values)
    
    def plot_trajectory_and_curvature(self, x, y, z, N, kappa, ax=None, color="b"):
        scale_factor = 0.1
        magnitudes = np.linalg.norm(N, axis=1)
        for i in range(len(x)):
            end_point = np.array([x[i], y[i], z[i]])
            curvature_direction = scale_factor * kappa[i] * N[i]
            ax.quiver(end_point[0], end_point[1], end_point[2],
                    curvature_direction[0], curvature_direction[1], curvature_direction[2]
                                    , color=color, length=magnitudes[i], normalize=True)

    def get_refined_forces(self, original_force1, original_force2, k, alpha=-1.0, beta=-1.0):
        return (original_force1/k)*alpha + (original_force2/k)*beta
        
    def getRefinedTrajecotry(self, example_id=1, noise_level=0.21, k=15):
        if example_id == 0 or example_id is None:
            example_id = 1

        kappa_min_max = [0, 1.5]
        tau_min_max = [-1, 1]
        view = []

        if example_id == 1:
            t = self.t_func([0, 2 * np.pi])
            r = 1
            x = r * np.sin(t)
            y = r * np.cos(t)
            z = r * np.cos(2 * t)
            
        if example_id == 2:
            t = self.t_func([0, 2 * np.pi])
            r = 1
            x = t
            y = r * np.cos(t)
            z = r * np.cos(2 * t)
            
        if example_id == 3:
            t = self.t_func([0, 2 * np.pi])
            r = 1
            x = t
            y = t
            z = r * np.cos(2 * t)

        dr = np.column_stack((np.gradient(x), np.gradient(y), np.gradient(z)))
        ds = np.linalg.norm(dr, axis=1)
        T = dr / ds[:, np.newaxis]
        dT = np.gradient(T, axis=0)
        dTds = dT / ds[:, np.newaxis]
        kappa = np.linalg.norm(dTds, axis=1)
        N = dTds / kappa[:, np.newaxis]
        B = np.cross(T, N)
        dB = np.gradient(B, axis=0)
        tau = -np.sum(dB * N, axis=1) / ds

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        force_mag = self.get_magnitude(np.sin, -0.2, 0.2, len(x))
        
        force_profile = (N.T*force_mag).T
        self.plot_trajectory_and_curvature(x, y, z, force_profile, kappa, ax=ax, color=(1, 0.0, 0.0, 0.1))
        
        force_profile_with_noise = self.add_noise_to_normal_vectors(force_profile, noise_level=noise_level)
        self.plot_trajectory_and_curvature(x, y, z, force_profile_with_noise, kappa, ax=ax, color=(0.0, 1.0, 0.0, 0.1))
        
        resultant_foruce = force_profile + force_profile_with_noise
        self.plot_trajectory_and_curvature(x, y, z, resultant_foruce, kappa
                                           , ax=ax, color=(0.0, 0.0, 1.0, 0.1))
        
        ax.plot(x, y, z, label='initial trajecotry', color="k")
        
        expected_forces = resultant_foruce/k
        expected_traj = np.vstack((x + expected_forces[:,0], y + expected_forces[:,1]
                                  , z + expected_forces[:,2])).T
        ax.plot(expected_traj[:,0], expected_traj[:,1], expected_traj[:,2]
                , label='expected trajecotry', color="y", linewidth=3.0)
        
        refined_forces = self.get_refined_forces(force_profile
                            , force_profile_with_noise, k, alpha=-1.0, beta=-1.0)
        
        refined_traj = np.vstack((x + refined_forces[:,0], y + refined_forces[:,1]
                                  , z + refined_forces[:,2])).T
        
        ax.plot(refined_traj[:,0], refined_traj[:,1], refined_traj[:,2]
                , label='refined trajecotry', color="magenta", linewidth=3.0)
        
        ax.set_title(f'With Noise (level={noise_level})')
        ax.set_title('Trajectory and Curvature Direction')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        ax.view_init(*view)

        plt.tight_layout()
        plt.show()

gen_force_profile = GenerateForceProfile()
gen_force_profile.getRefinedTrajecotry(example_id=3, noise_level=0.21, k=5)

