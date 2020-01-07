import numpy as np
from scipy.interpolate import RectBivariateSpline
from LucasKanade import interpolateImage
from LucasKanade import computeA
from LucasKanade import computeb
from LucasKanade import makeMeshGrid


def LucasKanadeBasis(It, It1, rect, bases):
    """
	Input:
		It: template image
		It1: Current image
		rect: Current position of the car
		(top left, bot right coordinates)
		bases: [n, m, k] where nxm is the size of the template.
	Output:
		p: movement vector [dp_x, dp_y]

    Put your implementation here
    """

    thre = 0.01
    p0 = np.zeros(2)
    p = np.zeros(2)
    bases = bases.reshape(-1, bases.shape[2])
    # print("bases = " + str(bases.shape))
    BBT = np.dot(bases, np.transpose(bases))
    B_multiplier = np.eye(BBT.shape[0]) - BBT

    It_interp = interpolateImage(It)
    It1_interp = interpolateImage(It1)

    p = p.reshape(2, 1) + p0.reshape(2, 1)
    delta_p = np.array([50, 50])

    x_mesh_flatten, y_mesh_flatten = makeMeshGrid(rect)

    while np.linalg.norm(delta_p) > thre:

        x_mesh_flatten_It1 = x_mesh_flatten + p[1]
        y_mesh_flatten_It1 = y_mesh_flatten + p[0]

        ori_A = computeA(y_mesh_flatten_It1, x_mesh_flatten_It1, It1_interp)

        ori_b = computeb(It_interp, It1_interp, y_mesh_flatten, x_mesh_flatten, y_mesh_flatten_It1, x_mesh_flatten_It1)

        A = np.dot(B_multiplier, ori_A)
        b = np.dot(B_multiplier, ori_b)

        delta_p = np.linalg.lstsq(A, b)[0]

        p = p.reshape(2, 1) + delta_p.reshape(2, 1)

    return p
