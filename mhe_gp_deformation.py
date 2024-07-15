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
# train_x_np = np.array([d_n_x, d_n_y, d_n_z, d_p_x, d_p_y, d_p_z]).T
train_y_np = np.array([m_deformed_traj_x, m_deformed_traj_y, m_deformed_traj_z]).T
train_x_np = np.linspace(0, 200, train_y_np.shape[0])
# train_y_np = np.array(m_deformed_traj_x)

# train_x = torch.from_numpy(train_x_np)
# Training data is 100 points in [0,1] inclusive regularly spaced
# train_x = torch.linspace(0, 1, 100)
# train_y = torch.sin(train_x * (2 * math.pi)) + torch.randn(train_x.size()) * math.sqrt(0.04)
train_y = torch.from_numpy(train_y_np)
train_y = train_y.type(torch.FloatTensor)
train_x = torch.from_numpy(train_x_np)
train_x = train_x.type(torch.FloatTensor)


likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=3)
model = MultitaskGPModel(train_x, train_y, likelihood)
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)  # Includes GaussianLikelihood parameters

checkpoint = torch.load('/home/op/fttraj/gp_deformation.pth')
model.load_state_dict(checkpoint['model_state_dict'])
likelihood.load_state_dict(checkpoint['likelihood_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

model.eval()
likelihood.eval()

# Initialize plots


# Make predictions
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    test_x = train_x
    predictions = likelihood(model(test_x))
    mean = predictions.mean
    lower, upper = predictions.confidence_region()


# MHE parameters
window_size = 20  # size of the sliding window
prediction_horizon = 20  # how many steps ahead we want to predict

# Initialize the sliding window (with initial data)
current_window_x = train_x[:window_size]
current_window_y = train_y[:window_size]

# Function to update the sliding window
def update_window(new_x, new_y, current_window_x, current_window_y, window_size):
    if len(current_window_x) >= window_size:
        current_window_x = current_window_x[1:]  # Remove the oldest data point
        current_window_y = current_window_y[1:]
    current_window_x = torch.cat([current_window_x, new_x.unsqueeze(0)], dim=0)
    current_window_y = torch.cat([current_window_y, new_y.unsqueeze(0)], dim=0)
    return current_window_x, current_window_y

# Prediction function
def predict_deformation(model, likelihood, current_window_x, prediction_horizon):
    model.eval()
    likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        # Generate prediction points within the prediction horizon
        prediction_points = current_window_x
        # Make predictions
        predictions = likelihood(model(prediction_points))
        mean = predictions.mean
        lower, upper = predictions.confidence_region()
        return mean, lower, upper


# Visualization function
def plot_predictions(time_points, mean, lower, upper, true_deformation, new_x, new_y): 
    
    f, (y1_ax, y2_ax, y3_ax) = plt.subplots(1, 3, figsize=(8, 3))
    y1_ax.plot(time_points, true_deformation.detach().numpy()[:,0], 'k*')
    y1_ax.plot(time_points.numpy(), mean[:, 0].numpy(), 'b')
    # y1_ax.plot(new_x.numpy(), new_y.numpy()[0], 'g*')
    # Shade in confidence
    y1_ax.fill_between(time_points.numpy(), lower[:, 0].numpy(), upper[:, 0].numpy(), alpha=0.5)
    y1_ax.set_ylim([-0.5, 3])
    y1_ax.legend(['actual', 'predict', 'confidence'])
    y1_ax.set_title('deformation in x-axis')
    y1_ax.set_xlabel('time (s)')
    y1_ax.set_ylabel('deformation (m)')

    
    # Predictive mean as blue line
    y2_ax.plot(time_points, true_deformation.detach().numpy()[:,1], 'k*')
    y2_ax.plot(time_points.numpy(), mean[:, 1].numpy(), 'b')
    # y2_ax.plot(new_x.numpy(), new_y.numpy()[1], 'g*')
    # Shade in confidence
    y2_ax.fill_between(time_points.numpy(), lower[:, 1].numpy(), upper[:, 1].numpy(), alpha=0.5)
    y2_ax.set_ylim([-0.5, 3])
    y2_ax.legend(['actual', 'predict', 'confidence'])
    y2_ax.set_title('deformation in y-axis')
    y2_ax.set_xlabel('time (s)')
    y2_ax.set_ylabel('deformation (m)')

    
    # Predictive mean as blue line
    y3_ax.plot(time_points, true_deformation.detach().numpy()[:,2], 'k*')
    y3_ax.plot(time_points.numpy(), mean[:, 2].numpy(), 'b')
    # y3_ax.plot(new_x.numpy(), new_y.numpy()[2], 'g*')
    # Shade in confidence
    y3_ax.fill_between(time_points.numpy(), lower[:, 2].numpy(), upper[:, 2].numpy(), alpha=0.5)
    y3_ax.set_ylim([-0.5, 2.5])
    y3_ax.legend(['actual', 'predict', 'confidence'])
    y3_ax.set_title('deformation in z-axis')
    y3_ax.set_xlabel('time (s)')
    y3_ax.set_ylabel('deformation (m)')
    plt.show()

# Initialize MHE process
for t in range(window_size, len(train_x) - prediction_horizon):
    # Get the new data point (from the training data for this example)
    new_x = train_x[t]
    new_y = train_y[t]
    
    # Update the sliding window
    current_window_x, current_window_y = update_window(new_x, new_y, current_window_x, current_window_y, window_size)
    
    # Predict future deformations
    mean, lower, upper = predict_deformation(model, likelihood, current_window_x, prediction_horizon)
    
    # True deformations for comparison (assuming train_y contains true values)
    true_deformation = train_y[t:t + prediction_horizon]
    
    # Output the predictions for this time step
    print(f"Time {t}: Mean predictions: {mean}")
    print(f"Time {t}: Confidence intervals: Lower: {lower}, Upper: {upper}")
    
    # Plot the predictions
    plot_predictions(current_window_x, mean, lower, upper, true_deformation, new_x, new_y)






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

