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
    
# Load the CSV file into a pandas DataFrame
df = pd.read_csv('/home/op/fttraj/new_data/data_31.csv')
df = df.dropna()
# df = self.df_ful[["f_x","f_y","f_z", "diff_x", "diff_y", "diff_z"]]

# Display the first few rows to understand the structure of the DataFrame
# print(df.head())

# Create a 3D plot
# ax = plt.figure().add_subplot(projection='3d')

start_index = 0
end_index = -1

n_traj_x, n_traj_y, n_traj_z = df['x'][start_index:end_index], df['y'][start_index:end_index], df['z'][start_index:end_index]
d_n_x, d_n_y, d_n_z = df['d_x'][start_index:end_index], df['d_y'][start_index:end_index], df['d_z'][start_index:end_index]
d_p_x, d_p_y, d_p_z = d_n_x, d_n_y, d_n_z

traj_dn_x, traj_dn_y, traj_dn_z = n_traj_x + d_n_x, n_traj_y + d_n_y, n_traj_z + d_n_z
traj_dp_x, traj_dp_y, traj_dp_z = n_traj_x + d_p_x, n_traj_y + d_p_y, n_traj_z + d_p_z

deformed_traj_x = n_traj_x + d_n_x + d_p_x
deformed_traj_y = n_traj_y + d_n_y + d_p_y
deformed_traj_z = n_traj_z + d_n_z + d_p_z 


mirror_deformed_traj_x = n_traj_x -( d_n_x + d_p_x )
mirror_deformed_traj_y = n_traj_y -( d_n_y + d_p_y )
mirror_deformed_traj_z = n_traj_z -( d_n_z + d_p_z )

m_deformed_traj_x = ( d_n_x + d_p_x )
m_deformed_traj_y = ( d_n_y + d_p_y )
m_deformed_traj_z = ( d_n_z + d_p_z )


# # # Training data is 100 points in [0,1] inclusive regularly spaced
# train_x_np = np.array([d_n_x, d_n_y, d_n_z]).T
# train_y_np = np.array([m_deformed_traj_x, m_deformed_traj_y, m_deformed_traj_z]).T
# train_x_np = np.linspace(0, 200, train_y_np.shape[0])
# train_y_np = np.array(m_deformed_traj_x)
train_x_np = np.array([n_traj_x, n_traj_y, n_traj_z]).T
train_y_np = np.array([d_n_x, d_n_y, d_n_z]).T

# train_x = torch.from_numpy(train_x_np)
# Training data is 100 points in [0,1] inclusive regularly spaced
# train_x = torch.linspace(0, 1, 100)
# train_y = torch.sin(train_x * (2 * math.pi)) + torch.randn(train_x.size()) * math.sqrt(0.04)
train_y = torch.from_numpy(train_y_np)
train_y = train_y.type(torch.FloatTensor)
train_x = torch.from_numpy(train_x_np)
train_x = train_x.type(torch.FloatTensor)


# ax = plt.figure().add_subplot(projection='3d')
# ax.plot(train_x_np, train_y_np[:,0])
# ax.plot(train_x_np, train_y_np[:,1])
# ax.plot(train_x_np, train_y_np[:,2])
# plt.show()


class MultitaskGPModelGRU(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, input_size, hidden_size, num_layers, num_tasks=2):
        super(MultitaskGPModelGRU, self).__init__(train_x, train_y, likelihood)
        
        # Define the GRU network
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        
        # Define a linear layer to map the GRU output to the appropriate size
        self.linear = nn.Linear(hidden_size, input_size)
        
        # Define the mean and covariance modules for GP
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=num_tasks
        )
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.RBFKernel(), num_tasks=num_tasks, rank=1
        )
        self.likelihood = likelihood

    def forward(self, x):
        # Pass the input through the GRU
        gru_out, _ = self.gru(x)
        
        # Apply the linear transformation
        x = self.linear(gru_out)  # Use the output of the last time step
        
        # Pass the transformed input through the GP model components
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)

    
    def predict(self, test_x):
        with torch.no_grad():
            # The output of the model is a multitask MVN, where both the data points
            # and the tasks are jointly distributed
            # To compute the marginal predictive NLL of each data point,
            # we will call `to_data_independent_dist`,
            # which removes the data cross-covariance terms from the distribution.
            preds = model.likelihood(model(test_x)).to_data_independent_dist()
        return preds.mean, preds.variance
    
training_iterations = 5000
num_tasks = 3
likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=num_tasks)
input_size=3
hidden_size=1
num_layers=5
model = MultitaskGPModelGRU(train_x=train_x, train_y=train_y
                        , likelihood=likelihood, input_size=input_size, hidden_size=hidden_size
                        , num_layers=num_layers, num_tasks=num_tasks)

# # # # find optimal model hyperparameters
# model.train()
# model.likelihood.train()
# model.to(device)

# # # # use the adam optimizer
# optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

# mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)


# train_x, train_y = train_x.to(device), train_y.to(device)
# test_x = train_x.to(device)
# # print(train_x)
# for i in tqdm(range(training_iterations), desc="Training"):
#     optimizer.zero_grad()
#     output = model(train_x)
#     loss = -mll(output, train_y)
#     # epochs_iter.set_postfix(loss=loss.item())
#     loss.backward()
#     print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iterations, loss.item()))
#     optimizer.step()
#     if(i%100 == 0):
#         torch.save({
#             'model_state_dict': model.state_dict(),
#             'likelihood_state_dict': likelihood.state_dict(),
#             'optimizer_state_dict': optimizer.state_dict()
#         }, '/home/op/fttraj/gp_deformation_'+str(i)+'_31.pth')

    

# Save the model and optimizer
checkpoint = torch.load('/home/op/fttraj/gp_deformation_4600_31.pth')

model.load_state_dict(checkpoint['model_state_dict'])
likelihood.load_state_dict(checkpoint['likelihood_state_dict'])

# Reinitialize the optimizer and load its state
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# Set model and likelihood to evaluation mode for inference
model.eval()
likelihood.eval()

with torch.no_grad(), gpytorch.settings.fast_pred_var():
    test_x = train_x
    predictions = likelihood(model(test_x))
    mean = predictions.mean
    lower, upper = predictions.confidence_region()

# # This contains predictions for both tasks, flattened out
# # The first half of the predictions is for the first task
# # The second half is for the second task


# # Initialize plots
f, (y1_ax, y2_ax, y3_ax) = plt.subplots(1, 3, figsize=(8, 3))
time_index = torch.linspace(0, 2000, test_x.cpu().numpy()[:,0].shape[0])
# Plot training data as black stars
y1_ax.plot(time_index, train_y[:, 0].cpu().numpy(), 'r')
# Predictive mean as blue line
y1_ax.plot(time_index, mean[:, 0].cpu().numpy(), 'b')
# Shade in confidence
# y1_ax.fill_between(time_index, lower[:, 0].cpu().numpy(), upper[:, 0].cpu().numpy(), alpha=0.2)
# y1_ax.set_ylim([-100, 100])
y1_ax.legend(['observed deformation', 'estimated deformation', 'confidence'])
y1_ax.set_title('deformation in x-axis')
y1_ax.set_xlabel('time (s)')
y1_ax.set_ylabel('deformation (m)')

y2_ax.plot(time_index, train_y[:, 1].cpu().numpy(), 'r')
# Predictive mean as blue line
y2_ax.plot(time_index, mean[:, 1].cpu().numpy(), 'b')
# Shade in confidence
# y2_ax.fill_between(time_index, mean[:, 1].cpu().numpy(), lower[:, 1].cpu().numpy(), upper[:, 1].cpu().numpy(), alpha=0.5)
# y2_ax.set_ylim([-0.5, 3])
y2_ax.legend(['observed deformation', 'estimated deformation', 'confidence'])
y2_ax.set_title('deformation in y-axis')
y2_ax.set_xlabel('time (s)')
y2_ax.set_ylabel('deformation (m)')

y3_ax.plot(time_index, train_y[:, 2].cpu().numpy(), 'r')
# Predictive mean as blue line
y3_ax.plot(time_index, mean[:, 2].cpu().numpy(), 'b')
# Shade in confidence
# y3_ax.fill_between(time_index, lower[:, 2].cpu().numpy(), upper[:, 2].cpu().numpy(), alpha=0.5)
# y3_ax.set_ylim([-0.5, 3])
y3_ax.legend(['observed deformation', 'estimated deformation', 'confidence'])
y3_ax.set_title('deformation in z-axis')
y3_ax.set_xlabel('time (s)')
y3_ax.set_ylabel('deformation (m)')

plt.show()

n_traj_x_projected, n_traj_y_projected, n_traj_z_projected = n_traj_x - mean[:,0].numpy(), n_traj_y - mean[:,1].numpy(), n_traj_z - mean[:,2].numpy()
fig = plt.figure(figsize=(14, 6))
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot(n_traj_x_projected, n_traj_y_projected, n_traj_z_projected, 'k', label="estimated trajectory")
ax1.set_xlabel('x (m)')
ax1.set_ylabel('y (m)')
ax1.set_zlabel('z (m)')
ax1.legend()

ax2 = fig.add_subplot(122, projection='3d')
ax2.plot(mirror_deformed_traj_x, mirror_deformed_traj_y, mirror_deformed_traj_z, 'r', label="expected trajectory")
ax2.set_xlabel('x (m)')
ax2.set_ylabel('y (m)')
ax2.set_zlabel('z (m)')
ax2.legend()
plt.show()

def_error_estimated = np.array([mean[:,0].cpu().numpy(), mean[:,1].cpu().numpy(), mean[:,2].cpu().numpy()])
def_actual = np.array([d_n_x, d_n_y, d_n_z])
diff = def_error_estimated - def_actual
# # Compute the magnitude of the differences
diff_magnitude = np.linalg.norm(diff, axis=0)
train_linespace = torch.linspace(0, diff_magnitude.shape[0], diff_magnitude.shape[0])

# # Plot the magnitude of the differences
plt.figure()
plt.plot(train_linespace, diff_magnitude, label='magnitude of differences', color='m')
plt.xlabel('time (s)')
plt.ylabel('magnitude of differences')
plt.title('magnitude of differences between trajectories (expected and estimated)')
plt.legend()
plt.show()

