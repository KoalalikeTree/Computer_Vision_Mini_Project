'''
Q4.2:
    1. Integrating everything together.
    2. Loads necessary files from ../data/ and visualizes 3D reconstruction using scatter
'''

'''
Q4.2:
    1. Integrating everything together.
    2. Loads necessary files from ../data/ and visualizes 3D reconstruction using scatter
'''
from submission import *
from helper import *

import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D


with np.load('../data/templeCoords.npz') as coordinate:
    x1 = coordinate['x1']
    n = x1.shape[0]
    y1 = coordinate['y1']

with np.load('../data/some_corresp.npz') as some_corresp:
    pts1 = some_corresp['pts1']
    pts2 = some_corresp['pts2']

with np.load('../data/intrinsics.npz') as intrinsics:
    K1 = intrinsics['K1']
    K2 = intrinsics['K2']

im1 = plt.imread('../data/im1.png')
im2 = plt.imread('../data/im2.png')

x2 = np.zeros((n, 1))
y2 = np.zeros((n, 1))

M = max(im1.shape)

F = eightpoint(pts1, pts2, M)
E = essentialMatrix(F, K1, K2)

for i in range(n):
    x2[i], y2[i] = epipolarCorrespondence(im1, im2, F, x1[i], y1[i])
pts_im1 = np.hstack((x1, y1))
pts_im2 = np.hstack((x2, y2))

M1 = np.diag([1.0, 1.0, 1.0])
M1 = np.hstack((M1, np.zeros((3, 1))))
M2_list = camera2(E)

C1 = np.dot(K1, M1)

least_err = 1e12
lestC2 = 0
leastP = 0
least_ind = 0

iter = 4
for i in range(iter):
    M2 = M2_list[:, :, i]
    C2 = np.dot(K2, M2)
    P, error = triangulate(C1, pts_im1, C2, pts_im2)
    if error < least_err:
        if np.min(P[:, 2] >= 0):
            least_err = error
            min_index = i
            lestC2 = C2
            leastP = P
np.savez('q4_2.npz', M1=M1, M2=M2_list[:, :, min_index], C1=C1, C2=lestC2)
print("find M2 error: ", least_err)
P = np.copy(leastP)
# print(P)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x_left_boundary = np.min(P[:, 0])
x_right_boundary = np.max(P[:, 0])
y_bottom_boundary= np.min(P[:, 1])
y_top_boundary = np.max(P[:, 1])
z_front_boudary = np.min(P[:, 2])
z_back_boudary = np.max(P[:, 2])


ax.set_xlim3d(x_left_boundary, x_right_boundary)
ax.set_ylim3d(y_bottom_boundary, y_top_boundary)
ax.set_zlim3d(z_front_boudary, z_back_boudary)

ax.scatter(P[:, 0], P[:, 1], P[:, 2], c='r', marker='o')
plt.show()
