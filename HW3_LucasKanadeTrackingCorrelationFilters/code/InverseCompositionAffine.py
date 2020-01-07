import numpy as np
from scipy.interpolate import RectBivariateSpline
from LucasKanade import interpolateImage
from LucasKanade import makeMeshGrid

def InverseCompositionAffine(It, It1):
	# Input: 
	#	It: template image
	#	It1: Current image

	# Output:
	#	M: the Affine warp matrix [2x3 numpy array]

    # put your implementation here
    M = np.eye(3).astype(np.double)

    p = np.zeros((6, 1)).astype(np.double)
    delta_p = np.ones((6, 1)).astype(np.double)

    # Interpolate
    It_interp = interpolateImage(It)
    It1_interp = interpolateImage(It1)

    w = It.shape[1]
    h = It.shape[0]
    rect = np.array([0, 0, w - 1, h - 1]).astype(np.double)
    x_mesh_It, y_mesh_It = makeMeshGrid(rect)

    # ▽T
    x_gradient_It = It_interp.ev(y_mesh_It, x_mesh_It, dx=1)
    y_gradient_It = It_interp.ev(y_mesh_It, x_mesh_It, dy=1)

    # ▽T * (σw/σp)
    A3 = (x_gradient_It * x_mesh_It).reshape(-1, 1)
    A4 = (x_gradient_It * y_mesh_It).reshape(-1, 1)
    A5 = x_gradient_It.reshape(-1, 1)
    A0 = (y_gradient_It * x_mesh_It).reshape(-1, 1)
    A1 = (y_gradient_It * y_mesh_It).reshape(-1, 1)
    A2 = y_gradient_It.reshape(-1, 1)
    A = np.hstack((A0, A1, A2, A3, A4, A5))

    # T(x)
    intensity_It = It_interp.ev(y_mesh_It, x_mesh_It).astype(np.double)
    It_point_coordinate = np.vstack(
        (x_mesh_It.reshape(1, -1), y_mesh_It.reshape(1, -1), np.ones((1, x_mesh_It.size))))

    thre = 0.01
    while np.linalg.norm(delta_p) > thre:
        # print(np.linalg.norm(delta_p))

        # W(x;p)
        It1_coordinate = np.dot(M, It_point_coordinate)

        # warp the coordinate of It1 in It frame back into It1 frame
        It_coordinate = np.dot(np.linalg.inv(M), It1_coordinate)
        # (3, wh)

        # find out which coordinate is included both in It and It1
        # filter out those point that x is not within the width of It
        filter_x = np.bitwise_and((It_coordinate[0, :] >= 0), (It_coordinate[0, :] < w))
        filter_y = np.bitwise_and((It_coordinate[1, :] >= 0), (It_coordinate[1, :] < h))
        filter_xy = np.bitwise_and(filter_x, filter_y)

        A_with_coordinate = np.transpose(A)
        A_with_coordinate = A_with_coordinate[:, np.where(filter_xy)].reshape(6, -1)
        A_with_coordinate = np.transpose(A_with_coordinate)

        # print("A_with_coordinate.shape" + str(A_with_coordinate.shape))
        It_coordinate = It_coordinate[:, np.where(filter_xy)].reshape(3, -1)

        x_coordinate_It = It_coordinate[0, :].astype(np.double)
        y_coordinate_It = It_coordinate[1, :].astype(np.double)

        # warp the qualified point back to the frame of It1
        It1_coordinate = np.dot(M, It_coordinate)
        x_coordinate_It1 = It1_coordinate[0, :].astype(np.double)
        y_coordinate_It1 = It1_coordinate[1, :].astype(np.double)

        # I(W(x;p))
        intensity_It1 = It1_interp.ev(y_coordinate_It1, x_coordinate_It1).astype(np.double)
        intensity_It = It_interp.ev(y_coordinate_It, x_coordinate_It).astype(np.double)

        # I(W(x;p)) - T(x)
        b = intensity_It1 - intensity_It
        # print("b.shape" + str(b.shape))

        # Compute Δp
        delta_p = np.linalg.lstsq(A_with_coordinate, b, rcond=None)[0]

        # W(x;p) ← W（x;p）⚪ W(x;Δp)^(-1)
        M_deltaP = np.eye(3).astype(np.double)
        M_deltaP[0, :] = [1.0 + delta_p[0], delta_p[1], delta_p[2]]
        M_deltaP[1, :] = [delta_p[3], 1.0 + delta_p[4], delta_p[5]]
        M = np.dot(M, np.linalg.inv(M_deltaP))

    return M[0:2, :]
