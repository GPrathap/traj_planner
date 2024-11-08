import math
import torch
import gpytorch
from matplotlib import pyplot as plt
import os
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 20})
from multi_gp import MultitaskGPModel
from mpl_toolkits.mplot3d import Axes3D

import numpy as np 
# this is for running the notebook in our testing framework
import os
import gpytorch
import os
import time
from matplotlib import cm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import os
import torch
from tqdm import tqdm
import math
import gpytorch
from torch.nn import Linear
from gpytorch.means import ConstantMean, LinearMean
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.variational import VariationalStrategy, CholeskyVariationalDistribution, \
    LMCVariationalStrategy
from gpytorch.distributions import MultivariateNormal
from gpytorch.models.deep_gps import DeepGPLayer, DeepGP
from gpytorch.mlls import DeepApproximateMLL, VariationalELBO
from gpytorch.likelihoods import MultitaskGaussianLikelihood
from matplotlib import pyplot as plt
from torch.cuda.amp import GradScaler, autocast

torch.cuda.empty_cache()
gpytorch.settings.cholesky_jitter(1e-4)


# torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
is_cuda = torch.cuda.is_available()

# If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
# if is_cuda:
#     device = torch.device("cuda")
#     print("GPU is available")
# else:
device = torch.device("cpu")
    
import numpy as np
import pandas as pd
import torch
import gpytorch
from torch import nn
import matplotlib.pyplot as plt

class DeformationTrajectoryAnalysis:
    def __init__(self, csv_path, input_size=3, hidden_size=1, num_layers=5, num_tasks=3):
        self.csv_path = csv_path
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_tasks = num_tasks
        self.likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=num_tasks)
        self.model = None
        self.train_x = None
        self.train_y = None
        
    def load_data(self):
        df = pd.read_csv(self.csv_path).dropna()
        start_index, end_index = 0, -1
        n_traj_x, n_traj_y, n_traj_z = df['x'][start_index:end_index], df['y'][start_index:end_index], df['z'][start_index:end_index]
        self.d_n_x, self.d_n_y, self.d_n_z = df['d_x'][start_index:end_index], df['d_y'][start_index:end_index], df['d_z'][start_index:end_index]

        # Prepare training data
        self.train_x = torch.FloatTensor(np.array([n_traj_x, n_traj_y, n_traj_z]).T)
        self.train_y = torch.FloatTensor(np.array([self.d_n_x, self.d_n_y, self.d_n_z]).T)
        
        d_p_x, d_p_y, d_p_z = self.d_n_x, self.d_n_y, self.d_n_z
        
        self.mirror_deformed_traj_x = n_traj_x -( self.d_n_x + d_p_x )
        self.mirror_deformed_traj_y = n_traj_y -( self.d_n_y + d_p_y )
        self.mirror_deformed_traj_z = n_traj_z -( self.d_n_z + d_p_z )

    def initialize_model(self):
        class MultitaskGPModelGRU(gpytorch.models.ExactGP):
            def __init__(self, train_x, train_y, likelihood, input_size, hidden_size, num_layers, num_tasks):
                super().__init__(train_x, train_y, likelihood)
                self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
                self.linear = nn.Linear(hidden_size, input_size)
                self.mean_module = gpytorch.means.MultitaskMean(gpytorch.means.ConstantMean(), num_tasks=num_tasks)
                self.covar_module = gpytorch.kernels.MultitaskKernel(gpytorch.kernels.RBFKernel(), num_tasks=num_tasks, rank=1)
                self.likelihood = likelihood

            def forward(self, x):
                gru_out, _ = self.gru(x)
                x = self.linear(gru_out)
                mean_x = self.mean_module(x)
                covar_x = self.covar_module(x)
                return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)
        
        self.model = MultitaskGPModelGRU(self.train_x, self.train_y, self.likelihood, self.input_size, self.hidden_size, self.num_layers, self.num_tasks)
    
    def train_model(self, training_iterations=5000, lr=0.0005):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
        self.model.train()
        self.likelihood.train()

        for i in range(training_iterations):
            optimizer.zero_grad()
            output = self.model(self.train_x)
            loss = -mll(output, self.train_y)
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                print(f'Iter {i + 1}/{training_iterations} - Loss: {loss.item()}')

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.likelihood.load_state_dict(checkpoint['likelihood_state_dict'])
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0005)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.model.eval()
        self.likelihood.eval()
        
    def get_predictions(self):
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            predictions = self.likelihood(self.model(self.train_x))
            self.mean = predictions.mean.cpu().numpy()
            self.lower, self.upper = predictions.confidence_region()
            def_error_estimated = np.array([self.mean[:,0], self.mean[:,1], self.mean[:,2]])
            def_actual = np.array([self.d_n_x, self.d_n_y, self.d_n_z])
            diff = def_error_estimated - def_actual
            self.diff_magnitude = np.linalg.norm(diff, axis=0)
            return self.mean, self.lower.cpu().numpy(), self.upper.cpu().numpy(), self.diff_magnitude
    
    def plot_predictions(self):
        
        f, (y1_ax, y2_ax, y3_ax) = plt.subplots(1, 3, figsize=(15, 5))
        time_index = torch.linspace(0, 2000, self.train_x.shape[0])

        for i, (ax, axis) in enumerate(zip([y1_ax, y2_ax, y3_ax], ['x', 'y', 'z'])):
            ax.plot(time_index, self.train_y[:, i].cpu().numpy(), 'r')
            ax.plot(time_index, self.mean[:, i].cpu().numpy(), 'b')
            # ax.fill_between(time_index, self.lower[:, i].cpu().numpy(), self.upper[:, i].cpu().numpy(), alpha=0.2)
            ax.legend(['Observed deformation', 'Estimated deformation', 'Confidence'])
            ax.set_title(f'Deformation in {axis}-axis')
            ax.set_xlabel('time (s)')
            ax.set_ylabel('deformation (m)')
            
            
        n_traj_x_projected, n_traj_y_projected, n_traj_z_projected = self.train_x[:,0] - self.mean[:,0].numpy(), self.train_x[:,1] - self.mean[:,1].numpy(), self.train_x[:,2] - self.mean[:,2].numpy()
        
        fig = plt.figure(figsize=(14, 6))
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.plot(n_traj_x_projected, n_traj_y_projected, n_traj_z_projected, 'k', label="estimated trajectory")
        ax1.set_xlabel('x (m)')
        ax1.set_ylabel('y (m)')
        ax1.set_zlabel('z (m)')
        ax1.legend()

        ax2 = fig.add_subplot(122, projection='3d')
        ax2.plot(self.mirror_deformed_traj_x, self.mirror_deformed_traj_y, self.mirror_deformed_traj_z, 'r', label="expected trajectory")
        ax2.set_xlabel('x (m)')
        ax2.set_ylabel('y (m)')
        ax2.set_zlabel('z (m)')
        ax2.legend()
        

        plt.figure(figsize=(10, 6))
        plt.boxplot(self.diff_magnitude, vert=False, patch_artist=True, boxprops=dict(facecolor="skyblue"))
        plt.xlabel('Magnitude of Differences')
        plt.title('Distribution of Magnitude of Differences Between Trajectories (Expected vs. Estimated)')

        train_linespace = torch.linspace(0, self.diff_magnitude.shape[0], self.diff_magnitude.shape[0])
        plt.figure()
        plt.plot(train_linespace, self.diff_magnitude, label='Magnitude of differences', color='m')
        plt.xlabel('time (s)')
        plt.ylabel('Magnitude of differences')
        plt.title('Magnitude of Differences Between Trajectories (Expected and Estimated)')
        plt.legend()
        plt.show()



# traning_data = '/home/op/fttraj/new_data/data_31.csv'
# check_point = '/home/op/fttraj/gp_deformation_4300_31.pth'

# modelling = DeformationTrajectoryAnalysis(traning_data)
# modelling.load_data()
# modelling.initialize_model()
# modelling.load_checkpoint(check_point)
# modelling.get_predictions()
# modelling.plot_predictions()


def plot_multiple_models(csv_paths, checkpoint_paths):
    models = []
    colors = ['b', 'g', 'm', 'c', 'y']  # Define colors for each model's plot lines
    
    for csv_path, checkpoint_path in zip(csv_paths, checkpoint_paths):
        analysis = DeformationTrajectoryAnalysis(csv_path)
        analysis.load_data()
        analysis.initialize_model()
        analysis.load_checkpoint(checkpoint_path)
        models.append(analysis)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    # fig2, axes2 = plt.subplots(1, 3, figsize=(18, 6))
    
    all_diff_magnitudes = []
    
    
    
    for i, model in enumerate(models):
        mean, lower, upper, error_mag = model.get_predictions()
        actual = model.train_y.cpu().numpy()
        all_diff_magnitudes.append(error_mag)
        for j, axis in enumerate(['x', 'y', 'z']):
            ax = axes[j]
            time_index = np.arange(actual[:, j].shape[0]) 
            ax.plot(time_index, actual[:, j], color='r', alpha=0.4 if i == 0 else 0, label="Observed" if i == 0 else "")
            ax.plot(time_index, mean[:, j], color=colors[i % len(colors)], label=f'Estimated Model {i+1}')
            # ax.fill_between(time_index, lower[:, j], upper[:, j], color=colors[i % len(colors)], alpha=0.1)
            ax.set_title(f'Deformation in {axis}-axis')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Deformation (m)')
    
    labels=['angle 25','angle 27','angle 29','angle 31','angle 33']
    axes[0].legend(loc="upper right")
    
    plt.figure(figsize=(10, 6))
    plt.boxplot(all_diff_magnitudes, labels=labels, vert=False, patch_artist=True, boxprops=dict(facecolor="skyblue"))
    plt.xlabel('Magnitude of Differences')
    plt.title('Distribution of Magnitude of Differences Between Trajectories (Expected vs. Estimated) for All Models')

    # plt.figure()
    # train_linespace = torch.linspace(0, all_diff_magnitudes[0].shape[0], all_diff_magnitudes[0].shape[0])
    # for dff_mag in all_diff_magnitudes:
    #     plt.plot(train_linespace, dff_mag, label='magnitude of differences')
    # plt.xlabel('time (s)')
    # plt.ylabel('magnitude of differences')
    # plt.title('magnitude of differences between trajectories (expected and estimated)')
    # plt.legend()
    
    plt.tight_layout()
    plt.show()

csv_paths = ['/home/op/fttraj/new_data/data_25.csv', '/home/op/fttraj/new_data/data_27.csv', '/home/op/fttraj/new_data/data_29.csv', '/home/op/fttraj/new_data/data_31.csv', '/home/op/fttraj/new_data/data_33.csv']
checkpoint_paths = ['/home/op/fttraj/gp_deformation_4600.pth', '/home/op/fttraj/gp_deformation_4600_27.pth'
                    ,  '/home/op/fttraj/gp_deformation_4600_29.pth',  '/home/op/fttraj/gp_deformation_4600_31.pth',  '/home/op/fttraj/new_data/model_33/gp_deformation_4600_31.pth']

plot_multiple_models(csv_paths, checkpoint_paths)




