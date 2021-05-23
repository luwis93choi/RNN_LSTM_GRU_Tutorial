import numpy as np

import matplotlib.pyplot as plt

VO_est_list = np.loadtxt('./CRNN_VO_est.txt').reshape(-1, 2)
VO_GT_list = np.loadtxt('./CRNN_groundtruth.txt').reshape(-1, 2)

# print(VO_est_list.shape)

GT_axis_max_lim = np.max(VO_GT_list)
est_axis_max_lim = np.max(VO_est_list)

axis_max_lim = np.divide(np.max(np.array([GT_axis_max_lim, est_axis_max_lim])), 100)

plt.title('Deep CRNN Visual Odometry Result')
plt.xlabel('X [m]')
plt.ylabel('Z [m]')
plt.xlim(-axis_max_lim, axis_max_lim)
plt.plot(np.divide(VO_est_list[:, 0], 100), np.divide(VO_est_list[:, 1], 100) , color='b', label='VO Pose Estimation')
plt.plot(np.divide(VO_GT_list[:, 0], 100), np.divide(VO_GT_list[:, 1], 100) , color='r', label='VO Groundtruth')
plt.legend()
plt.show()