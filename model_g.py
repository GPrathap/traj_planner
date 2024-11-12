import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
plt.rcParams.update({'font.size': 18})
# Load the CSV file into a pandas DataFrame
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


m_deformed_traj_x = n_traj_x -( d_n_x + d_p_x )
m_deformed_traj_y = n_traj_y -( d_n_y + d_p_y )
m_deformed_traj_z = n_traj_z -( d_n_z + d_p_z )


fig = plt.figure(figsize=(14, 6))
ax1 = fig.add_subplot(131, projection='3d')
ax1.plot(n_traj_x, n_traj_y, n_traj_z, label='desired trajectory')
ax1.set_xlabel('x (mm)')
ax1.set_ylabel('y (mm)')
ax1.set_zlabel('z (mm)')
ax1.legend()

ax2 = fig.add_subplot(132, projection='3d')
ax2.plot(deformed_traj_x, deformed_traj_y, deformed_traj_z, label='deformed trajectory')
ax2.set_xlabel('x (mm)')
ax2.set_ylabel('y (mm)')
ax2.set_zlabel('z (mm)')
ax2.legend()

ax3 = fig.add_subplot(133, projection='3d')
ax3.plot(m_deformed_traj_x, m_deformed_traj_y, m_deformed_traj_z, label='refined trajectory')
ax3.set_xlabel('x (mm)')
ax3.set_ylabel('y (mm)')
ax3.set_zlabel('z (mm)')
ax3.legend()
# ax.plot(traj_dn_x, traj_dn_y, traj_dn_z, label='trajectory with material deformation')
# ax.plot(traj_dp_x, traj_dp_y, traj_dp_z, label='trajectory with spring deformation')  

# Add labels and title



# Show the plot
plt.show()
