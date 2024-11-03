import math
import torch
import gpytorch
from matplotlib import pyplot as plt
import os
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 20})
from multi_gp import MultitaskGPModel
# from mpl_toolkits.mplot3d import Axes3D
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


mirror_deformed_traj_x = n_traj_x -( d_n_x + d_p_x )
mirror_deformed_traj_y = n_traj_y -( d_n_y + d_p_y )
mirror_deformed_traj_z = n_traj_z -( d_n_z + d_p_z )

m_deformed_traj_x = ( d_n_x + d_p_x )
m_deformed_traj_y = ( d_n_y + d_p_y )
m_deformed_traj_z = ( d_n_z + d_p_z )


# # # Training data is 100 points in [0,1] inclusive regularly spaced
train_x_np = np.array([n_traj_x, n_traj_y, n_traj_z]).T
train_y_np = np.array([d_n_x, d_n_y, d_n_z]).T
# train_x_np = np.linspace(0, 200, train_y_np.shape[0])
# train_y_np = np.array(m_deformed_traj_x)

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



likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=3)
model = MultitaskGPModel(train_x, train_y, likelihood)


smoke_test = ('CI' in os.environ)
training_iterations = 2 if smoke_test else 50


# Find optimal model hyperparameters
model.train()
likelihood.train()

# Use the adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)  # Includes GaussianLikelihood parameters

# "Loss" for GPs - the marginal log likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

for i in range(training_iterations):
    optimizer.zero_grad()
    output = model(train_x)
    loss = -mll(output, train_y)
    loss.backward()
    print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iterations, loss.item()))
    optimizer.step()
    
# Set into eval mode
model.eval()
likelihood.eval()

# # Initialize plots
# f, (y1_ax, y2_ax, y3_ax) = plt.subplots(1, 3, figsize=(8, 3))


# # Save the model and optimizer
# torch.save({
#     'model_state_dict': model.state_dict(),
#     'likelihood_state_dict': likelihood.state_dict(),
#     'optimizer_state_dict': optimizer.state_dict()
# }, '/home/op/fttraj/gp_deformation.pth')

# # Load the model and optimizer
# checkpoint = torch.load('/home/op/fttraj/gp_deformation.pth')
# model.load_state_dict(checkpoint['model_state_dict'])
# likelihood.load_state_dict(checkpoint['likelihood_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# model.eval()
# likelihood.eval()
# # state_dict = torch.load('/home/op/fttraj/gp_deformation.pth')
# # likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=3)
# # model = MultitaskGPModel(train_x, train_y, likelihood)
# # model.load_state_dict(state_dict)

# # Make predictions
# with torch.no_grad(), gpytorch.settings.fast_pred_var():
#     test_x = train_x
#     predictions = likelihood(model(test_x))
#     mean = predictions.mean
#     lower, upper = predictions.confidence_region()

# # This contains predictions for both tasks, flattened out
# # The first half of the predictions is for the first task
# # The second half is for the second task



# # Plot training data as black stars
# y1_ax.plot(train_x.detach().numpy(), train_y[:, 0].detach().numpy(), 'k*')
# # Predictive mean as blue line
# y1_ax.plot(test_x.numpy(), mean[:, 0].numpy(), 'b')
# # Shade in confidence
# y1_ax.fill_between(test_x.numpy(), lower[:, 0].numpy(), upper[:, 0].numpy(), alpha=0.5)
# y1_ax.set_ylim([-0.5, 3])
# y1_ax.legend(['observed data', 'mean', 'confidence'])
# y1_ax.set_title('deformation in x-axis')
# y1_ax.set_xlabel('time (s)')
# y1_ax.set_ylabel('deformation (m)')

# # Plot training data as black stars
# y2_ax.plot(train_x.detach().numpy(), train_y[:, 1].detach().numpy(), 'k*')
# # Predictive mean as blue line
# y2_ax.plot(test_x.numpy(), mean[:, 1].numpy(), 'b')
# # Shade in confidence
# y2_ax.fill_between(test_x.numpy(), lower[:, 1].numpy(), upper[:, 1].numpy(), alpha=0.5)
# y2_ax.set_ylim([-0.5, 3])
# y2_ax.legend(['observed data', 'mean', 'confidence'])
# y2_ax.set_title('deformation in y-axis')
# y2_ax.set_xlabel('time (s)')
# y2_ax.set_ylabel('deformation (m)')

# y3_ax.plot(train_x.detach().numpy(), train_y[:, 2].detach().numpy(), 'k*')
# # Predictive mean as blue line
# y3_ax.plot(test_x.numpy(), mean[:, 2].numpy(), 'b')
# # Shade in confidence
# y3_ax.fill_between(test_x.numpy(), lower[:, 2].numpy(), upper[:, 2].numpy(), alpha=0.5)
# y3_ax.set_ylim([-0.5, 2.5])
# y3_ax.legend(['observed data', 'mean', 'confidence'])
# y3_ax.set_title('deformation in z-axis')
# y3_ax.set_xlabel('time (s)')
# y3_ax.set_ylabel('deformation (m)')

# n_traj_x_projected, n_traj_y_projected, n_traj_z_projected = n_traj_x - mean[:,0].numpy(), n_traj_y - mean[:,1].numpy(), n_traj_z - mean[:,2].numpy()

# fig = plt.figure(figsize=(14, 6))
# ax1 = fig.add_subplot(121, projection='3d')
# ax1.plot(n_traj_x_projected, n_traj_y_projected, n_traj_z_projected, 'k', label="estimated trajectory")
# ax1.set_xlabel('x (m)')
# ax1.set_ylabel('y (m)')
# ax1.set_zlabel('z (m)')
# ax1.legend()

# # Add second subplot for the differences
# ax2 = fig.add_subplot(122, projection='3d')
# ax2.plot(mirror_deformed_traj_x, mirror_deformed_traj_y, mirror_deformed_traj_z, 'b', label="expected trajectory")
# ax2.set_xlabel('x (m)')
# ax2.set_ylabel('y (m)')
# ax2.set_zlabel('z (m)')
# ax2.legend()

# # ax3 = fig.add_subplot(133, projection='3d')
# # ax3.plot(mirror_deformed_traj_x, mirror_deformed_traj_y, mirror_deformed_traj_z, 'b', label="expected trajectory")
# # ax3.plot(n_traj_x_projected, n_traj_y_projected, n_traj_z_projected, 'k', label="estimated trajectory")
# # ax3.set_xlabel('x (m)')
# # ax3.set_ylabel('y (m)')
# # ax3.set_zlabel('z (m)')
# # ax3.legend()

# plt.legend()

# diff_x = n_traj_x_projected-mirror_deformed_traj_x
# diff_y = n_traj_y_projected-mirror_deformed_traj_y
# diff_z = n_traj_z_projected-mirror_deformed_traj_z

# # Compute the magnitude of the differences
# diff_magnitude = np.sqrt(diff_x**2 + diff_y**2 + diff_z**2)

# # Plot the magnitude of the differences
# plt.figure()
# plt.plot(train_x_np, diff_magnitude, label='magnitude of differences', color='m')
# plt.xlabel('time (s)')
# plt.ylabel('magnitude of differences')
# plt.title('magnitude of differences between trajectories (expected and estimated)')
# plt.legend()
# plt.show()

