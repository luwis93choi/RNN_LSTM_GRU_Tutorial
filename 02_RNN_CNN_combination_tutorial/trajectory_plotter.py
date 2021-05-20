import numpy as np

import matplotlib.pyplot as plt

VO_est_list = np.loadtxt('./CRNN_VO_est.txt').reshape(-1, 2)
VO_GT_list = np.loadtxt('./CRNN_groundtruth.txt').reshape(-1, 2)

# print(VO_est_list.shape)

plt.title('Deep CRNN Visual Odometry Result')
plt.xlabel('X [cm]')
plt.ylabel('Z [cm]')
plt.plot(VO_est_list[:, 0], VO_est_list[:, 1] , color='b', label='VO Pose Estimation')
plt.plot(VO_GT_list[:, 0], VO_GT_list[:, 1] , color='r', label='VO Groundtruth')
plt.legend()
plt.show()