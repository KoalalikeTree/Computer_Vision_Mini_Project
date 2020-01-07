import numpy as np
from scipy import ndimage
from scipy.interpolate import RectBivariateSpline
from LucasKanade import interpolateImage
from LucasKanade import computeb
from LucasKanade import makeMeshGrid


def LucasKanadeAffine(It, It1):
	# Input: 
	#	It: template image
	#	It1: Current image
	# Output:
	#	M: the Affine warp matrix [2x3 numpy array]
    # put your implementation here

    # Initialize M as an (3, 3) identity matrix
    M = np.eye(3).astype(np.double)

    # Initialize p and delta p to be (6, 1) vector
    p = np.zeros((6, 1)).astype(np.double)
    delta_p = np.ones((6, 1)).astype(np.double)

    # Set the ROI rect cover the whole image
    w = It.shape[1]
    h = It.shape[0]
    rect = np.array([0, 0, w-1, h-1]).astype(np.double)

    # (1, wh)
    x_mesh_flatten, y_mesh_flatten = makeMeshGrid(rect)

    It1_interp = interpolateImage(It1)
    It_interp = interpolateImage(It)

    # make a new (3, wh) It coordinate to be affined into It1 coordinate
    It1_point_coordinate = np.vstack((x_mesh_flatten.reshape(1, -1), y_mesh_flatten.reshape(1, -1), np.ones((1, x_mesh_flatten.size))))

    thre = 0.1
    while np.linalg.norm(delta_p) >= thre:
        print(np.linalg.norm(delta_p))

        # warp the coordinate of It1 in It frame back into It1 frame
        It_coordinate = np.dot(np.linalg.inv(M), It1_point_coordinate)
        # (3, wh)

        # find out which coordinate is included both in It and It1
        # filter out those point that x is not within the width of It
        filter_x = np.bitwise_and((It_coordinate[0, :] >= 0), (It_coordinate[0, :] < w))
        filter_y = np.bitwise_and((It_coordinate[1, :] >= 0), (It_coordinate[1, :] < h))
        filter_xy = np.bitwise_and(filter_x, filter_y)
        It_coordinate = It_coordinate[:, np.where(filter_xy)].reshape(3, -1)

        # warp the qualified point back to the frame of It1
        It1_coordinate = np.dot(M, It_coordinate)

        # compute A
        x_coordinate_It1 = It1_coordinate[0, :].astype(np.double)
        y_coordinate_It1 = It1_coordinate[1, :].astype(np.double)

        x_gradient_It1 = It1_interp.ev(y_coordinate_It1, x_coordinate_It1, dx=1).reshape(-1, 1)
        y_gradient_It1 = It1_interp.ev(y_coordinate_It1, x_coordinate_It1, dy=1).reshape(-1, 1)

        A0 = x_gradient_It1 * (x_coordinate_It1.reshape(x_coordinate_It1.size, 1))
        A1 = x_gradient_It1 * (y_coordinate_It1.reshape(y_coordinate_It1.size, 1))
        A2 = x_gradient_It1
        A3 = y_gradient_It1 * (x_coordinate_It1.reshape(x_coordinate_It1.size, 1))
        A4 = y_gradient_It1 * (y_coordinate_It1.reshape(y_coordinate_It1.size, 1))
        A5 = y_gradient_It1

        A = np.hstack((A0, A1, A2, A3, A4, A5))

        # compute b
        x_coordinate_It = It_coordinate[0, :].astype(np.double)
        y_coordinate_It = It_coordinate[1, :].astype(np.double)
        Intensity_It = It_interp.ev(y_coordinate_It, x_coordinate_It)
        Intensity_It1 = It1_interp.ev(y_coordinate_It1, x_coordinate_It1)
        b = Intensity_It - Intensity_It1

        delta_p = np.linalg.lstsq(A, b, rcond=None)[0]
        print("delta_p" + str(delta_p))

        p = p.reshape(6, 1) + delta_p.reshape(6, 1)

        M[0, :] = [1.0 + p[0], p[1], p[2]]
        M[1, :] = [p[3], 1.0 + p[4], p[5]]

    return M[0:2, :]




