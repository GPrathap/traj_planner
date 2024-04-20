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

    def plot_trajectory_and_curvature(self, x, y, z, T, N, B, kappa, tau, kappa_min_max
                                    , tau_min_max, view, ax=None, color="b", length=0.1):
        scale_factor = 0.1
        for i in range(len(x)):
            end_point = np.array([x[i], y[i], z[i]])
            curvature_direction = scale_factor * kappa[i] * N[i]
            ax.quiver(end_point[0], end_point[1], end_point[2],
                    curvature_direction[0], curvature_direction[1], curvature_direction[2]
                                    , color=color, length=length, normalize=True)

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


        # Calculate TNB
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

        # fig, axes = plt.subplots(1, 2, figsize=(16, 8), subplot_kw={'projection': '3d'})
        # ax = fig.add_subplot(111, projection='3d')
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        force_profile = N
        self.plot_trajectory_and_curvature(x, y, z, T, force_profile, B, kappa, tau, kappa_min_max
                                    , tau_min_max, view, ax=ax, color="r", length=0.1)

        force_profile_with_noise = self.add_noise_to_normal_vectors(N, noise_level=noise_level)
        self.plot_trajectory_and_curvature(x, y, z, T, force_profile_with_noise
                                    , B, kappa, tau, kappa_min_max, tau_min_max, view, ax=ax
                                    , color="b", length=0.1)
        
        resultant_foruce = force_profile + force_profile_with_noise
        self.plot_trajectory_and_curvature(x, y, z, T, resultant_foruce
                                    , B, kappa, tau, kappa_min_max, tau_min_max, view, ax=ax
                                    , color="g", length=0.1)
        
        sub_traj = resultant_foruce/k
        
        ax.plot(x, y, z, label='initial trajecotry', color="k")
        ax.plot(x + sub_traj[:,0], y + sub_traj[:,1], z + sub_traj[:,2], label='refined trajecotry', color="y")
        
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
gen_force_profile.getRefinedTrajecotry(example_id=1, noise_level=0.21, k=15)

