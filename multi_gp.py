
import gpytorch
import os
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from tqdm import tqdm 
from sklearn.preprocessing import MinMaxScaler
import os
import torch
import tqdm
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


# torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
is_cuda = torch.cuda.is_available()

# If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
if is_cuda:
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    
    
class DataProcessing:
    def __init__(self, dataset_file, inputs_cols_indices, label_col_index, max_num_points, train_test_ratio=0.1):
        self.label_col_index = label_col_index  
        self.output_data_size = len(label_col_index) 
        self.inputs_cols_indices = inputs_cols_indices     
        self.get_training_and_testing_dataset(dataset_file, train_test_ratio, max_num_points)
        
    def get_training_and_testing_dataset(self, dataset_file, train_test_ratio, max_num_points):
        if(max_num_points < 0.0):
            self.df_ful = pd.read_csv(dataset_file, skiprows=1)
        else:
            self.df_ful = pd.read_csv(dataset_file, skiprows=1, nrows=max_num_points)
        
        self.df = self.df_ful[["f_x","f_y","f_z", "def_dx", "def_dy", "def_dz"]]
        self.sc = MinMaxScaler()
        self.label_sc = MinMaxScaler()
        self.data = self.sc.fit_transform(self.df.values)
        # self.label = self.label_sc.fit(self.df.iloc[:, self.label_col_index].values.reshape(-1, self.output_data_size))
        self.inputs = self.data[:, self.inputs_cols_indices]
        self.labels = self.data[:, self.label_col_index]
        self.desired_traj = self.df_ful[["p_x", "p_y", "p_z"]].values
        self.desired_forces = self.df_ful[["f_x", "f_y", "f_z"]].values
        
        test_portion = int(train_test_ratio * len(self.inputs))
        self.train_x = self.inputs[:-test_portion]
        self.train_y = self.labels[:-test_portion]
        self.desired_traj_training = self.desired_traj[:-test_portion]
        self.test_x = self.inputs[-test_portion:]
        self.test_y = self.labels[-test_portion:]
        self.desired_traj_testing = self.desired_traj[-test_portion:]
        
        print("======dataset: inputs: {} labels {}========".format(self.inputs.shape, self.labels.shape))
        print("training set: train_x: {} train_y {}".format(self.train_x.shape, self.train_y.shape))
        print("testing set: testing_x: {} testing_y {}".format(self.test_x.shape, self.test_y.shape))
    
    def get_ref_traj(self, ):
        return self.desired_traj
    
    def get_ref_forces(self, ):
        return self.desired_forces
    
    def get_training_data(self, is_torch=True):
        self.train_x = self.train_x.astype(np.float32)
        self.train_y = self.train_y.astype(np.float32)
        if(is_torch):
            return torch.from_numpy(self.train_x), torch.from_numpy(self.train_y), self.desired_traj_training
        return self.train_x, self.train_y, self.desired_traj_training
    
    def get_testing_data(self, is_torch=True):
        self.test_x = self.test_x.astype(np.float32)
        self.test_y = self.test_y.astype(np.float32)
        if(is_torch):
            return torch.from_numpy(self.test_x), torch.from_numpy(self.test_y), self.desired_traj_testing 
        return self.test_x, self.test_y, self.desired_traj_testing  
    
    
class VisModeling:
    def __init__(self):
        pass
    
    def plot_trajectory(self, ax, ref_traj, direction_vector, color="r", plot_force_profile=True):
        # plot the trajectory and curvature direction vectors
        x, y, z = ref_traj[:,0], ref_traj[:,1], ref_traj[:,2]
        ax.plot(x, y, z)
        if(plot_force_profile):
            magnitudes = np.linalg.norm(direction_vector, axis=1, keepdims=True)
            mag_max = np.max(magnitudes)
            direction_vector = direction_vector/magnitudes
            for i in range(len(x)):
                end_point = np.array([x[i], y[i], z[i]])
                curvature_direction = direction_vector[i]
                vec_mag = magnitudes[i]/mag_max
                ax.quiver(end_point[0], end_point[1], end_point[2], curvature_direction[0], curvature_direction[1], curvature_direction[2], color=color, length=vec_mag[0], normalize=True)

        
    
class MultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, num_tasks=2):
        super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=num_tasks
        )
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.RBFKernel(), num_tasks=num_tasks, rank=1
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)

class MultitaskGPModelGRU(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, input_size, hidden_size, num_layers, num_tasks=2):
        super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)
        
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

    def forward(self, x):
        # Pass the input through the GRU
        gru_out, _ = self.gru(x)
        
        # Apply the linear transformation
        x = self.linear(gru_out)  # Use the output of the last time step
        
        # Pass the transformed input through the GP model components
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)

class DGPHiddenLayer(DeepGPLayer):
    def __init__(self, input_dims, output_dims, num_inducing=128, linear_mean=True):
        inducing_points = torch.randn(output_dims, num_inducing, input_dims)
        batch_shape = torch.Size([output_dims])

        variational_distribution = CholeskyVariationalDistribution(
            num_inducing_points=num_inducing,
            batch_shape=batch_shape
        )
        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True
        )

        super().__init__(variational_strategy, input_dims, output_dims)
        self.mean_module = ConstantMean() if linear_mean else LinearMean(input_dims)
        self.covar_module = ScaleKernel(
            MaternKernel(nu=2.5, batch_shape=batch_shape, ard_num_dims=input_dims),
            batch_shape=batch_shape, ard_num_dims=None
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)
    
class MultitaskDeepGP(DeepGP):
    def __init__(self, train_x_shape, num_tasks, num_hidden_dgp_dims=3):
        hidden_layer = DGPHiddenLayer(
            input_dims=train_x_shape[-1],
            output_dims=num_hidden_dgp_dims,
            linear_mean=True
        )
        last_layer = DGPHiddenLayer(
            input_dims=hidden_layer.output_dims,
            output_dims=num_tasks,
            linear_mean=False
        )

        super().__init__()

        self.hidden_layer = hidden_layer
        self.last_layer = last_layer

        # We're going to use a ultitask likelihood instead of the standard GaussianLikelihood
        self.likelihood = MultitaskGaussianLikelihood(num_tasks=num_tasks)

    def forward(self, inputs):
        hidden_rep1 = self.hidden_layer(inputs)
        output = self.last_layer(hidden_rep1)
        return output

    def predict(self, test_x):
        with torch.no_grad():

            # The output of the model is a multitask MVN, where both the data points
            # and the tasks are jointly distributed
            # To compute the marginal predictive NLL of each data point,
            # we will call `to_data_independent_dist`,
            # which removes the data cross-covariance terms from the distribution.
            preds = model.likelihood(model(test_x)).to_data_independent_dist()

        return preds.mean.mean(0), preds.variance.mean(0)




data_file = "/home/op/fttraj/data/cone_profile.csv"
inputs_cols_indices = [0, 1, 2]
label_col_index = [3, 4, 5]
num_tasks = len(label_col_index)
max_num_points = 400
data_loader = DataProcessing(data_file, inputs_cols_indices, label_col_index, max_num_points, train_test_ratio=0.7)
train_x, train_y, desired_traj_training = data_loader.get_training_data()
test_x, test_y, desired_traj_testing = data_loader.get_testing_data()

training_iterations = 100
likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=num_tasks)


# model = MultitaskGPModel(train_x, train_y, likelihood, num_tasks=num_tasks)

# input_size=len(inputs_cols_indices)
# hidden_size=2
# num_layers=2
# model = MultitaskGPModelGRU(train_x=train_x, train_y=train_y, likelihood=likelihood
#                         , input_size=input_size, hidden_size=hidden_size
#                         , num_layers=num_layers, num_tasks=num_tasks)

model = MultitaskDeepGP(train_x.shape, num_tasks=num_tasks)
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
mll = DeepApproximateMLL(VariationalELBO(model.likelihood, model, num_data=train_y.size(0)))

num_epochs = 200

for i in range(num_epochs):
    optimizer.zero_grad()
    output = model(train_x.to(device).float())
    loss = -mll(output, train_y.to(device).float())
    # epochs_iter.set_postfix(loss=loss.item())
    loss.backward()
    optimizer.step()
    
model.eval()
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    mean, var = model.predict(test_x.to(device).float())
    lower = mean - 2 * var.sqrt()
    upper = mean + 2 * var.sqrt()

# Plot results
points = mean.numpy()
time_index = torch.linspace(0, 200, points.shape[0])
fig, axs = plt.subplots(1, num_tasks, figsize=(4 * num_tasks, 3))
for task, ax in enumerate(axs):
    # ax.plot(train_x.squeeze(-1).detach().numpy(), train_y[:, task].detach().numpy(), 'k*')
    ax.plot(time_index, mean[:, task].numpy(), 'b', label="estimated d"+str(task))
    ax.plot(time_index, test_x[:, task].numpy(), 'r', label="ref d"+str(task))
    ax.fill_between(time_index, lower[:, task].numpy(), upper[:, task].numpy(), alpha=0.5)
    ax.set_ylim([-3, 3])
    ax.legend()
    ax.set_title(f'Task {task + 1}')
fig.tight_layout()





# # find optimal model hyperparameters
# model.train()
# likelihood.train()

# # use the adam optimizer
# optimizer = torch.optim.Adam(model.parameters(), lr=0.05)  # Includes GaussianLikelihood parameters

# # "loss" for GPs - the marginal log likelihood
# mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

# for i in range(training_iterations):
#     optimizer.zero_grad()
#     output = model(train_x)
#     loss = -mll(output, train_y)
#     loss.backward()
#     print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iterations, loss.item()))
#     optimizer.step()
    
# # set into eval mode
# model.eval()
# likelihood.eval()


# # set into eval mode
# model.eval()
# likelihood.eval()



# # make predictions

# with torch.no_grad(), gpytorch.settings.fast_pred_var():
#     predictions = likelihood(model(test_x))
#     mean = predictions.mean
#     lower, upper = predictions.confidence_region()


# # plot training data as black stars
# ref_traj = data_loader.get_ref_traj()
    
# ref_forces = data_loader.get_ref_forces()

# vis_modeling = VisModeling()

# # fig = plt.figure(figsize=(10, 8))
# # ax = fig.add_subplot(111, projection='3d') 
# # vis_modeling.plot_trajectory(ax, ref_traj, ref_forces)

# # updated_traj = desired_traj_testing + mean.numpy()
# # fig = plt.figure(figsize=(10, 8))
# # ax = fig.add_subplot(111, projection='3d') 
# # ax.plot(desired_traj_testing[:,0], desired_traj_testing[:,1], desired_traj_testing[:,2], label="ref traj")
# # ax.plot(updated_traj[:,0], updated_traj[:,1], updated_traj[:,2], label="estimated traj")
# # ax.legend()

# # # # initialize plots
# f, (y1_ax, y2_ax, y3_ax) = plt.subplots(1, 3, figsize=(8, 3))
# # # predictive mean as blue line
# estimated = mean.numpy()
# print("number of testing data", estimated.shape[0])
# time_index = torch.linspace(0, 200, estimated.shape[0])

# index = 0
# y1_ax.plot(time_index, mean[:, index].numpy(), 'b', label="expected dx")
# y1_ax.plot(time_index, test_x[:, index].numpy(), 'r', label="ref dx")
# # y1_ax.fill_between(time_index, lower[:, index].numpy(), upper[:, index].numpy(), alpha=0.5)
# # y1_ax.set_ylim([-2, 2])

# index = 1
# y2_ax.plot(time_index, mean[:, index].numpy(), 'b', label="expected dy")
# y2_ax.plot(time_index, test_x[:, index].numpy(), 'r', label="ref dy")
# # y2_ax.fill_between(time_index, lower[:, index].numpy(), upper[:, index].numpy(), alpha=0.5)
# # y2_ax.set_ylim([-2, 2])

# index = 2
# y3_ax.plot(time_index, mean[:, index].numpy(), 'b', label="expected dz")
# y3_ax.plot(time_index, test_x[:, index].numpy(), 'r', label="ref dz")
# # y3_ax.fill_between(time_index, lower[:, index].numpy(), upper[:, index].numpy(), alpha=0.5)
# # y3_ax.set_ylim([-3, 3])

plt.show()