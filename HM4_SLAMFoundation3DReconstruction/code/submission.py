"""
Homework4.
Replace 'pass' by your implementation.
"""

# Insert your package here
import numpy as np
import numpy.matlib as matlib
import matplotlib.pyplot as plt
import sympy as sp
import scipy
from scipy.ndimage.filters import gaussian_filter
from math import *
from helper import *


def eightpoint(pts1, pts2, M):
    '''
    Q2.1: Eight Point Algorithm
        Input:  pts1, Nx2 Matrix
                pts2, Nx2 Matrix
                M, a scalar parameter computed as max (imwidth, imheight)
        Output: F, the fundamental matrix
    '''
    # Replace pass by your implementation
    """

    :param pts1: Nx2 Matrix
    :param pts2: pts2, Nx2 Matrix
    :param M: a scalar parameter computed as max (imwidth, imheight)
    :return:  F, the fundamental matrix
    """

    n = pts1.shape[0]

    # scale the data by dividing each coordinate by M
    pts1 = np.divide(pts1, M)
    pts2 = np.divide(pts2, M)

    # A is a N by 9 matrix
    # every row is insist of
    # xi2xi1; xi2yi1; xi2; yi2xi1; yi2yi1; yi2; xi1; yi1; 1
    A = np.ones((n, 9))
    A[:, 0] = pts2[:, 0] * pts1[:, 0]
    A[:, 1] = pts2[:, 0] * pts1[:, 1]
    A[:, 2] = pts2[:, 0]
    A[:, 3] = pts2[:, 1] * pts1[:, 0]
    A[:, 4] = pts2[:, 1] * pts1[:, 1]
    A[:, 5] = pts2[:, 1]
    A[:, 6] = pts1[:, 0]
    A[:, 7] = pts1[:, 1]

    # Get the last column as the solution of Af = 0
    # f = [f11 f12 f13 f21 f22 f23 f31 f32 f33]
    U, S, VT = np.linalg.svd(A)
    V = VT.transpose()
    F_vec = V[:, -1]
    F = F_vec.reshape((3, 3))

    # re-introduce the singularity constraint by taking the singular decomposition of F
    # setting the last singular value to zero, and multiplying the terms back out
    UF, SF, VTF = np.linalg.svd(F)
    SF[2] = 0
    SF = np.diag([SF[0], SF[1], SF[2]])
    F = np.dot(np.dot(UF, SF), VTF)

    # refine the solution by using local minimization
    F = refineF(F, pts1, pts2)

    # Unscaling
    # can't understand why the scaling matrix is like this
    scale = np.diag([1.0 / M, 1.0 / M, 1.0])
    F = np.dot(np.dot(scale.transpose(), F), scale)
    np.savez('q2_1.npz', F=F, M=M)

    print(F)

    return F


def sevenpoint(pts1, pts2, M):
    '''
    Q2.2: Seven Point Algorithm
        Input:  pts1, Nx2 Matrix
                pts2, Nx2 Matrix
                M, a scalar parameter computed as max (imwidth, imheight)
        Output: Farray, a list of estimated fundamental matrix.
    '''
    # Replace pass by your implementation
    pass

    n = pts1.shape[0]

    pts1 = np.divide(pts1, np.matlib.repmat(M, n, 2))
    pts2 = np.divide(pts2, np.matlib.repmat(M, n, 2))

    A = np.zeros((n, 9))
    A[:, 0] = pts2[:, 0] * pts1[:, 0]
    A[:, 1] = pts2[:, 0] * pts1[:, 1]
    A[:, 2] = pts2[:, 0]
    A[:, 3] = pts2[:, 1] * pts1[:, 0]
    A[:, 4] = pts2[:, 1] * pts1[:, 1]
    A[:, 5] = pts2[:, 1]
    A[:, 6] = pts1[:, 0]
    A[:, 7] = pts1[:, 1]
    A[:, 8] = np.ones(n)

    U, S, VT = np.linalg.svd(A)
    V = VT.transpose
    F1 = V()[:, -1]
    F1 = F1.reshape((3, 3))
    F2 = V()[:, -2]
    F2 = F2.reshape((3, 3))

    fun = lambda alpha: np.linalg.det(alpha * F1 + (1 - alpha) * F2)

    alpha0 = fun(0)
    alpha1 = 2.0 * (fun(1) - fun(-1)) / 3 - (fun(2) - fun(-2)) / 12
    alpha2 = 0.5 * fun(1) + 0.5 * fun(-1) - fun(0)
    alpha3 = fun(1) - alpha0 - alpha1 - alpha2
    roots = np.roots(np.array([alpha3, alpha2, alpha1, alpha0]))

    scale = np.diag([1.0 / M, 1.0 / M, 1.0])

    Fs = []
    for alpha in roots:
        F = F1 * float(np.real(alpha)) + F2 * (1 - float(np.real(alpha)))
        U, S, VT = np.linalg.svd(F)
        sigma = np.diag([S[0], S[1], S[2]])
        F = np.dot(np.dot(U, sigma), VT)
        F = np.dot(np.dot(scale.transpose(), F), scale)
        Fs.append(F)

    return Fs


def essentialMatrix(F, K1, K2):
    '''
    Q3.1: Compute the essential matrix E.
        Input:  F, fundamental matrix
                K1, internal camera calibration matrix of camera 1
                K2, internal camera calibration matrix of camera 2
        Output: E, the essential matrix
    '''
    # Replace pass by your implementation
    F = np.dot(K2.transpose(), F)
    F = np.dot(F, K1)
    return F


def skew(vector):
    """
    this function returns a numpy array with the skew symmetric cross product matrix for vector.
    the skew symmetric cross product matrix is defined such that
    np.cross(a, b) = np.dot(skew(a), b)

    :param vector: An array like vector to create the skew symmetric cross product matrix for
    :return: A numpy array of the skew symmetric cross product vector
    """

    return np.array([[0, -vector[2], vector[1]],
                     [vector[2], 0, -vector[0]],
                     [-vector[1], vector[0], 0]])


def triangulate(C1, pts1, C2, pts2):
    '''
    Q3.2: Triangulate a set of 2D coordinates in the image to a set of 3D points.
        Input:  C1, the 3x4 camera matrix
                pts1, the Nx2 matrix with the 2D image coordinates per row
                C2, the 3x4 camera matrix
                pts2, the Nx2 matrix with the 2D image coordinates per row
        Output: P, the Nx3 matrix with the corresponding 3D points per row
                err, the reprojection error.
    '''
    n = pts1.shape[0]

    P = np.zeros((n, 3))
    A = np.zeros((4, 4))
    for i in range(n):
        # compute A
        # methodology in https://blog.csdn.net/u011178262/article/details/86729887

        A[0, :] = pts1[i, 0] * C1[2, :] - C1[0, :]
        A[1, :] = pts1[i, 1] * C1[2, :] - C1[1, :]
        A[2, :] = pts2[i, 0] * C2[2, :] - C2[0, :]
        A[3, :] = pts2[i, 1] * C2[2, :] - C2[1, :]

        # solve P from AP = 0
        U, S, VT = np.linalg.svd(A)
        V = VT.transpose()
        p = V[:, -1]

        # normalize P
        P[i, :] = np.divide(p[0:3], p[3])

    # P in homogeneous presentation 4*100
    homoP = np.hstack((P, np.ones((n, 1))))
    homoP = np.transpose(homoP)

    # reproject P to camera 1 and camera 2
    reprojP1 = np.dot(C1, homoP)
    reprojP2 = np.dot(C2, homoP)

    normP1 = np.zeros((2, n))
    normP2 = np.zeros((2, n))
    normP1[0:2, :] = np.divide(reprojP1[0:2, :], reprojP1[2, :])
    normP2[0:2, :] = np.divide(reprojP2[0:2, :], reprojP2[2, :])

    # 100 * 3
    normP1 = normP1.transpose()
    normP2 = normP2.transpose()

    errP1 = (normP1 - pts1)[:, 0] ** 2 + (normP1 - pts1)[:, 1] ** 2
    errP2 = (normP2 - pts2)[:, 0] ** 2 + (normP2 - pts2)[:, 1] ** 2
    error = np.sum(errP1) + np.sum(errP2)

    return P, error


def epipolarCorrespondence(im1, im2, F, x1, y1):
    '''
    Q4.1: 3D visualization of the temple images.
        Input:  im1, the first image
                im2, the second image
                F, the fundamental matrix
                x1, x-coordinates of the pixel on im1
                y1, y-coordinates of the pixel on im1
        Output: x2, x-coordinates of the pixel on im2
                y2, y-coordinates of the pixel on im2

    '''

    # set window, bigger window less noise, smaller window more detail
    win = 10
    h = im2.shape[0]
    w = im2.shape[1]
    im1 = im1.astype(float)
    im2 = im2.astype(float)

    X1_homo = np.hstack((x1, y1, 1))
    F_X1 = np.dot(F, X1_homo)
    F_X1 = F_X1 / np.linalg.norm(F_X1)

    pts_within = np.empty((0, 2))

    # check if the window is in the boundary of the image
    a = F_X1[0]
    b = F_X1[1]
    c = F_X1[2]

    if a != 0:
        # if the epipolar line is horizontal
        for y in range(win, h - win):
            # calculate the x from the F_X1 constrain
            x = np.floor(-1.0 * (b * y + c) / a)
            if win <= x <= w - win:
                pts_within = np.append(pts_within, np.array([x, y]).reshape(1, 2), axis=0)

    pts = pts_within
    n = pts.shape[0]

    patch_im1 = im1[int(y1 - win + 1): int(y1 + win), int(x1 - win + 1): int(x1 + win), :]

    min_error = 1e12
    pt_with_min_erro_ind = 0

    for i in range(n):
        x2 = pts[i, 0]
        y2 = pts[i, 1]
        distance = sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        if distance < 50:
            patch_im2 = im2[int(y2 - win + 1): int(y2 + win), int(x2 - win + 1): int(x2 + win), :]
            err = patch_im1 - patch_im2
            gaussianErr = np.sum(gaussian_filter(err, sigma=1.0))
            if gaussianErr < min_error:
                min_error = gaussianErr
                pt_with_min_erro_ind = i

    x2 = pts[pt_with_min_erro_ind, 0]
    y2 = pts[pt_with_min_erro_ind, 1]

    return x2, y2


def ransacF(pts1, pts2, M):
    '''
    Q5.1: RANSAC method.
        Input:  pts1, Nx2 Matrix
                pts2, Nx2 Matrix
                M, a scaler parameter
        Output: F, the fundamental matrix
                inliers, Nx1 bool vector set to true for inliers
    '''
    err_threshold = 0.01

    iter_num = 500
    num_total_pts = pts1.shape[0]

    pts1_homo = np.hstack((pts1, np.ones((num_total_pts, 1))))
    pts2_homo = np.hstack((pts1, np.ones((num_total_pts, 1))))

    pts1_seven = np.zeros((7, 2))
    pts2_seven = np.copy(pts1_seven)

    chosen_7_pts_img1 = np.zeros((7, 2))
    chosen_7_pts_img2 = np.zeros((7, 2))

    max_inline_num = 0

    ultimate_chosen_pts_im1 = []
    ultimate_chosen_pts_im2 = []
    ultimate_chosen_pts_index = []

    for i in range(iter_num):

        # randomly choice 7 points from the dataset
        ind_chosen_7_pts = np.random.choice(n, 7)

        # get the chosen points from the original points
        for j in range(7):
            chosen_7_pts_img1[j] = pts1[ind_chosen_7_pts[j]]
            chosen_7_pts_img2[j] = pts2[ind_chosen_7_pts[j]]

        # compute fundamental matrix from the current im1 and im2
        Farray = sevenpoint(chosen_7_pts_img1, chosen_7_pts_img2, M)

        inline_pts_index = []
        for f_7pt in Farray:
            inline_pts_im1 = np.empty((0, 2))
            inline_pts_im2 = np.empty((0, 2))
            num_inline_pts = 0

            #judge if the point is inline
            for h in range(n):
                x1 = pts1_homo[h, :]
                x2 = pts2_homo[h, :]
                x2T = np.transpose(x2)
                x2TF = np.dot(x2T, f_7pt)
                x2TFx1 = np.dot(x2TF, x1)
                err = abs(x2TFx1)

                # inline points
                if err < err_threshold:
                    num_inline_pts += 1
                    inline_pts_im1 = np.append(inline_pts_im1, pts1[h, :].reshape(1, 2), axis=0)
                    inline_pts_im2 = np.append(inline_pts_im2, pts1[h, :].reshape(1, 2), axis=0)
                    inline_pts_index = np.append(inline_pts_index, h)


            if num_inline_pts > max_inline_num:
                ultimate_chosen_pts_im1 = inline_pts_im1
                ultimate_chosen_pts_im2 = inline_pts_im2
                ultimate_chosen_pts_index = np.array(inline_pts_index)
                max_inline_num = num_inline_pts
                print("Max num inliers: ", max_inline_num)
                print("inliers_index_final shape: ", ultimate_chosen_pts_index.shape)

    inliers = np.zeros((n, 1), dtype=bool)

    for i in ultimate_chosen_pts_index:
        inliers[i.astype(int)] = 1

    F = eightpoint(ultimate_chosen_pts_im1, ultimate_chosen_pts_im2, M)

    return F, inliers


'''
Q5.2: Rodrigues formula.
    Input:  r, a 3x1 vector
    Output: R, a rotation matrix
'''
def rodrigues(r):
    # Replace pass by your implementation
    def S(n):
        Sn = np.array([[0, -n[2], n[1]], [n[2], 0, -n[0]], [-n[1], n[0], 0]])
        return Sn

    theta = np.linalg.norm(r)
    if theta > 1e-30:
        n = r / theta
        Sn = S(n)
        R = np.eye(3) + sin(theta) * Sn + (1 - cos(theta)) * np.dot(Sn, Sn)
    else:
        Sr = S(r)
        theta2 = theta ** 2
        R = np.eye(3) + (1 - theta2 / 6.) * Sr + (.5 - theta2 / 24.) * np.dot(Sr, Sr)
    return np.mat(R)


def invRodrigues(R):
    '''
    Q5.2: Inverse Rodrigues formula.
        Input:  R, a rotation matrix
        Output: r, a 3x1 vector
    '''
    # Replace pass by your implementation
    epsilon = 1e-16
    theta = acos((np.trace(R) - 1) / 2.0)
    r = np.zeros((3, 1))

    if abs(theta) > epsilon:
        sin_mul_skew = np.divide((R - np.transpose(R)), 2)
        skew = sin_mul_skew/sin(theta)
        r = np.array([skew[2, 1], skew[0, 2], skew[1, 0]])
        r = r * theta
    return r


def rodriguesResidual(K1, M1, p1, K2, p2, x):
    '''
    Q5.3: Rodrigues residual.
        Input:  K1, the intrinsics of camera 1
                M1, the extrinsics of camera 1
                p1, the 2D coordinates of points in image 1
                K2, the intrinsics of camera 2
                p2, the 2D coordinates of points in image 2
                x, the flattened concatenationg of P, r2, and t2.
                where P = 1by3 3D coordinate
                      r2 = 1by3 rotation vector
                      t2 = 1by3 translation vector
        Output: residuals, 4N x 1 vector, the difference between original and estimated projections
    '''

    # Replace pass by your implementation
    P_3d = x[0:-6].reshape(-1, 3)
    rotation_vector = x[-6:-3].reshape(3, 1)
    translation = x[-3:].reshape(3, 1)

    rotationMatrix = rodrigues(rotation_vector)
    extrinsic = np.hstack((rotationMatrix, translation))

    cameraMatrix1 = np.dot(K1, M1)
    cameraMatrix2 = np.dot(K2, extrinsic)

    num_pts = P_3d.shape[0]
    P_homo = np.transpose(np.hstack((P_3d, np.ones((num_pts, 1)))))

    reproP1_2d = np.dot(cameraMatrix1, P_homo)
    reproP2_2d = np.dot(cameraMatrix2, P_homo)


    norm_P1_2d = np.transpose(np.divide(reproP1_2d, reproP1_2d[2, :])[0:2])
    norm_P2_2d = np.transpose(np.divide(reproP2_2d, reproP2_2d[2, :])[0:2])

    error1 = (p1 - norm_P1_2d).reshape(-1)
    error2 = (p2 - norm_P2_2d).reshape(-1)

    residuals = np.vstack((error1, error2))

    return residuals


def bundleAdjustment(K1, M1, p1, K2, M2_init, p2, P_init):
    '''
    Q5.3 Bundle adjustment.
        Input:  K1, the intrinsics of camera 1
                M1, the extrinsics of camera 1
                p1, the 2D coordinates of points in image 1
                K2,  the intrinsics of camera 2
                M2_init, the initial extrinsics of camera 1 3*4
                p2, the 2D coordinates of points in image 2
                P_init, the initial 3D coordinates of points
        Output: M2, the optimized extrinsics of camera 1
                P2, the optimized 3D coordinates of points
    '''

    # Replace pass by your implementation
    residual = lambda x: rodriguesResidual(K1, M1, p1, K2, p2, x)

    RotationMatrix2 = M2_init[:, 0:3]
    translation2 = M2_init[:, 3]
    rotationVector2 = invRodrigues(RotationMatrix2).reshape(-1)

    n_pts = P_init.shape[0]
    x = np.zeros(3 * n_pts + 6)
    x[-6:-3] = rotationVector2
    x[-3:] = translation2
    x[0:-6] = P_init.reshape(-1)

    optimize_x, = scipy.optimize.leastsq(residual, x)
    optimized_rotationvector = optimize_x[-6:-3].reshape(3, 1)
    optimized_translation = optimize_x[-3:]
    P = optimize_x[0:-6].reshape(-1, 3)

    optimized_RotationMatrix = rodrigues(optimized_rotationvector)
    M2 = np.hstack((optimized_RotationMatrix, optimized_translation.reshape(3, 1)))

    return M2, P


if __name__ == '__main__':
    with np.load('../data/some_corresp.npz') as some_corresp:
        pts1 = some_corresp['pts1']
        pts2 = some_corresp['pts2']
    # points number
    n = max(pts1.shape)
    with np.load('../data/intrinsics.npz') as intrinsics:
        K1 = intrinsics['K1']
        K2 = intrinsics['K2']

    with np.load('../data/some_corresp_noisy.npz') as data_noisy:
        pts1_noisy = data_noisy['pts1']
        pts2_noisy = data_noisy['pts2']

    im1 = plt.imread('../data/im1.png')
    im2 = plt.imread('../data/im2.png')
    M = max(im1.shape)

    # # test Q2.1
    #
    # F = eightpoint(pts1, pts2, M)
    # # displayEpipolarF(im1, im2, F)
    # print('Eight Points: F = ', F)
    #
    # # test Q2.2
    # indexes = np.random.choice(n, 7)
    #
    # p1 = np.zeros((7, 2))
    # p2 = np.zeros((7, 2))
    # for i in range(indexes.shape[0]):
    #     p1[i] = pts1[indexes[i]]
    #     p2[i] = pts2[indexes[i]]
    #
    # Farray = sevenpoint(p1, p2, M)
    # np.savez('q2_2.npz', F=Farray, M=M, pts1=p1, pts2=p2)
    # for f in Farray:
    #     print('seven points F = ', f)
    #     # displayEpipolarF(im1, im2, f)
    #
    #
    # # test Q3.1
    # M = essentialMatrix(F, K1, K2)
    # print('F reconstruct from M = ', M)
    #
    # # test Q4.2
    # M = max(im1.shape)
    # F = eightpoint(pts1, pts2, M)
    # epipolarMatchGUI(im1, im2, F)
    # np.savez('q4_1.npz', F=F, pts1=pts1, pts2=pts2)

    # test Q5.1
    F, inliers = ransacF(pts1_noisy, pts2_noisy, M)
    # displayEpipolarF(im1, im2, F)

    # test Q5.3
    pts1_in = np.empty((0, 2))
    pts2_in = np.empty((0, 2))

    for i in range(inliers.shape[0]):
        if inliers[i] == True:
            pts1_in = np.append(pts1_in, pts1_noisy[i].reshape(1, 2), axis=0)
            pts2_in = np.append(pts2_in, pts2_noisy[i].reshape(1, 2), axis=0)
    print("Num inliers: ", pts1_in.shape[0])
    F = refineF(F, pts1_in, pts2_in)

    E = essentialMatrix(F, K1, K2)
    M1 = np.array([[1.0, 0, 0, 0],
                   [0, 1.0, 0, 0],
                   [0, 0, 1.0, 0]])
    M2s = camera2(E)
    C1 = np.dot(K1, M1)

    min_error = 1e12
    min_M2 = 0
    min_C2 = 0
    min_P = 0
    min_index = 0

    for i in range(4):
        C2 = np.dot(K2, M2s[:, :, i])
        P, error = triangulate(C1, pts1_in, C2, pts2_in)
        if error < min_error:
            if np.min(P[:, 2] >= 0):
                # print("Found!")
                min_error = error
                min_index = i
                min_M2 = M2s[:, :, min_index]
                min_C2 = C2
                min_P = P

    M2 = np.copy(min_M2)
    C2 = np.copy(min_C2)
    P = np.copy(min_P)
    print("Error before bundleAdjustment: ", min_error)

    M2_ba, P_ba = bundleAdjustment(K1, M1, pts1_in, K2, M2, pts2_in, P)
    C2_ba = np.dot(K2, M2_ba)
    P_ba, error = triangulate(C1, pts1_in, C2_ba, pts2_in)
    print("Error after bundleAdjustment: ", error)
    # print(P.shape)
    # print(P_ba.shape)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    xmin1, xmax1 = np.min(P_ba[:, 0]), np.max(P_ba[:, 0])
    ymin1, ymax1 = np.min(P_ba[:, 1]), np.max(P_ba[:, 1])
    zmin1, zmax1 = np.min(P_ba[:, 2]), np.max(P_ba[:, 2])
    xmin2, xmax2 = np.min(P[:, 0]), np.max(P[:, 0])
    ymin2, ymax2 = np.min(P[:, 1]), np.max(P[:, 1])
    zmin2, zmax2 = np.min(P[:, 2]), np.max(P[:, 2])

    xmin, xmax = min(xmin1, xmin2), max(xmax1, xmax2)
    ymin, ymax = min(ymin1, ymin2), max(ymax1, ymax2)
    zmin, zmax = min(zmin1, zmin2), max(zmax1, zmax2)
    # xmin, xmax = -1, 1
    # ymin, ymax = -0.5, 0.5
    # zmin, zmax = 0, 2

    ax.set_xlim3d(xmin, xmax)
    ax.set_ylim3d(ymin, ymax)
    ax.set_zlim3d(zmin, zmax)

    ax.scatter(P_ba[:, 0], P_ba[:, 1], P_ba[:, 2], c='r', marker='o')
    ax.scatter(P[:, 0], P[:, 1], P[:, 2], c='b', marker='o')
    plt.show()
