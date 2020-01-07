import numpy as np
import cv2
import os
from planarH import computeH
from matplotlib import pyplot as plt


def compute_extrinsics(K, H):
    '''
    INPUTS:
        K - intrinsic parameters matrix
        H - estimated homography
    OUTPUTS:
        R - relative 3D rotation
        t - relative 3D translation
    '''

    #############################
    # TO DO ...
    R = np.zeros((3, 3))
    lam = 0

    # 1. eliminate the intrinsic matrix in homography
    H_without_intri = np.matmul(np.linalg.inv(K), H)

    # 2. SVD with the first two column of  H(without intrinsic),
    # then calculate the fist two column of the Rotation Matrix
    U, _, VT = np.linalg.svd(H_without_intri[:, 0:2])
    R[:, 0:2] = -np.matmul(np.matmul(U, np.array([[1, 0], [0, 1], [0, 0]])), VT)

    # 3. calculate the third column of the rotation matrix
    # by cross dot of the first and second column of the rotation matrix
    R[:, 2] = np.cross(R[:, 0], R[:, 1]).reshape(3,)

    # 4. check if the det of the R == -1
    # multiple the third column by -1
    if np.linalg.det(R) == -1:
        R[:, 2] = R[:, 2] * -1

    print(np.linalg.det(R))
    # 5. calculate the lambda to normalize the translation vector
    lam = np.sum(np.divide(H_without_intri[:, 0:2], R[:, 0:2],)) / 6

    t = H_without_intri[:, 2]/lam

    return R, t


def project_extrinsics(K, W, R, t):

    Rt = np.append(R, t.reshape(3,1), axis=1)
    H = np.matmul(K, Rt)
    X = np.matmul(H, W)
    X_norm = (X[0:2, :] / X[2, :]).astype(int)

    return X_norm





if __name__ == "__main__":
    # image
    im = cv2.imread('../data/prince_book.jpeg')

    #############################
    # TO DO ...
    # W is the 3D planar corner of the book
    W = np.array([[0.0, 18.2, 18.2, 0.0],
                  [0.0, 0.0,  26.0, 26.0],
                  [0.0, 0.0,  0.0,  0.0]])

    # X is the projection of the corner
    X = np.array([[483, 1704, 2175, 67],
                  [810, 781,  2217, 2286]])

    # K is the intrinsic matrix of the camera
    K = np.array([[3043.72, 0.0,      1196.00],
                  [0.0,     3043.72,  1604.00],
                  [0.0,     0.0,      1.0]])

    # 0. Read the txt file as the 3D point of the tennis
    tennis = np.loadtxt('../data/sphere.txt')
    tennis = np.append(tennis, np.ones((1, tennis.shape[1])), axis=0)

    # 1. Compute homography matrix H based on given 3D points W and its projection X
    H2to1 = computeH(X, W[0:2])

    # 2. Calculate the Rotation matrix R and translation vector t
    # based on the intrinsic matrix k and homagraphy matrix H
    R, t = compute_extrinsics(K, H2to1)

    # 3. Calculate the 2D projection of the tennis points use intrinsic K, extrinsic R,t
    shift = np.array([[350], [820]])
    tennis_2d = project_extrinsics(K, tennis, R, t) + shift

    # 4. visualization
    fig = plt.figure()
    plt.imshow(im)

    for i in range(tennis_2d.shape[1]):
        plt.plot(tennis_2d[0, i], tennis_2d[1, i], 'y.', markersize=1)
    plt.draw()
    plt.show()

    print(1)


    # perform required operations and plot sphere
