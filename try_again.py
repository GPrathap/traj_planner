import math
import torch
import gpytorch
from matplotlib import pyplot as plt
import os
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np 
# this is for running the notebook in our testing framework
import os
# Load the CSV file into a pandas DataFrame
df = pd.read_csv('/home/op/fttraj/cone_profile.csv',skiprows=1)


# Display the first few rows to understand the structure of the DataFrame
# print(df.head())

# Create a 3D plot
# ax = plt.figure().add_subplot(projection='3d')

start_index = 0
end_index = -1

n_traj_x, n_traj_y, n_traj_z = df['p_x'][start_index:end_index], df['p_y'][start_index:end_index], df['p_z'][start_index:end_index]
d_n_x, d_n_y, d_n_z = df['def_dx'][start_index:end_index], df['def_dy'][start_index:end_index], df['def_dz'][start_index:end_index]
d_p_x, d_p_y, d_p_z = df['spring_dx'][start_index:end_index], df['spring_dy'][start_index:end_index], df['spring_dz'][start_index:end_index]

traj_dn_x, traj_dn_y, traj_dn_z = n_traj_x + d_n_x, n_traj_y + d_n_y, n_traj_z + d_n_z
traj_dp_x, traj_dp_y, traj_dp_z = n_traj_x + d_p_x, n_traj_y + d_p_y, n_traj_z + d_p_z

deformed_traj_x = n_traj_x + d_n_x + d_p_x
deformed_traj_y = n_traj_y + d_n_y + d_p_y
deformed_traj_z = n_traj_z + d_n_z + d_p_z 


m_deformed_traj_x = n_traj_x -( d_n_x + d_p_x )
m_deformed_traj_y = n_traj_y -( d_n_y + d_p_y )
m_deformed_traj_z = n_traj_z -( d_n_z + d_p_z )


# # # Training data is 100 points in [0,1] inclusive regularly spaced
# train_x_np = np.array([d_n_x, d_n_y, d_n_z, d_p_x, d_p_y, d_p_z]).T
train_y_np = np.array(n_traj_x)
# train_y_np = np.array(m_deformed_traj_x)

# train_x = torch.from_numpy(train_x_np)
# Training data is 100 points in [0,1] inclusive regularly spaced
# train_x = torch.linspace(0, 1, 100)
# train_y = torch.sin(train_x * (2 * math.pi)) + torch.randn(train_x.size()) * math.sqrt(0.04)
train_y = torch.from_numpy(train_y_np)
train_y = train_y.type(torch.FloatTensor)
train_x = torch.linspace(0, 1000, train_y_np.shape[0])


# # Training data is 100 points in [0,1] inclusive regularly spaced
# train_x = torch.linspace(0, 1, 100)
# # True function is sin(2*pi*x) with Gaussian noise
# train_y = torch.sin(train_x * (2 * math.pi)) + torch.randn(train_x.size()) * math.sqrt(0.04)

# We will use the simplest form of GP model, exact inference
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# initialize likelihood and model
likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = ExactGPModel(train_x, train_y, likelihood)

# this is for running the notebook in our testing framework
import os
smoke_test = ('CI' in os.environ)
training_iter = 2 if smoke_test else 50


# Find optimal model hyperparameters
model.train()
likelihood.train()

# Use the adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

# "Loss" for GPs - the marginal log likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

for i in range(training_iter):
    # Zero gradients from previous iteration
    optimizer.zero_grad()
    # Output from model
    output = model(train_x)
    # Calc loss and backprop gradients
    loss = -mll(output, train_y)
    loss.backward()
    print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
        i + 1, training_iter, loss.item(),
        model.covar_module.base_kernel.lengthscale.item(),
        model.likelihood.noise.item()
    ))
    optimizer.step()

# Get into evaluation (predictive posterior) mode
model.eval()
likelihood.eval()

# Test points are regularly spaced along [0,1]
# Make predictions by feeding model through likelihood
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    test_x = train_x
    observed_pred = likelihood(model(test_x))
    
with torch.no_grad():
    # Initialize plot
    f, ax = plt.subplots(1, 1, figsize=(4, 3))

    # Get upper and lower confidence bounds
    lower, upper = observed_pred.confidence_region()
    # Plot training data as black stars
    ax.plot(train_x.numpy(), train_y.numpy(), 'k*')
    # Plot predictive means as blue line
    ax.plot(test_x.numpy(), observed_pred.mean.numpy(), 'b')
    # Shade between the lower and upper confidence bounds
    ax.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
    ax.set_ylim([-30, 30])
    ax.legend(['Observed Data', 'Mean', 'Confidence'])
    
plt.show()