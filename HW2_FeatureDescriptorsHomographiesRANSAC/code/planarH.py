import numpy as np
from numpy import matlib
import cv2
from BRIEF import briefLite, briefMatch, plotMatches, makeTestPattern


def computeH(p1, p2):
    '''
    INPUTS:
        p1 and p2 - Each are size (2 x N) matrices of corresponding (x, y)'
                 coordinates between two images
    OUTPUTS:
     H2to1 - a 3 x 3 matrix encoding the homography that best matches the linear
            equation
    '''
    assert (p1.shape[1] == p2.shape[1])
    assert (p1.shape[0] == 2)
    #############################
    # TO DO ...
    # append ones to bottom row of p1, p2 so [x,y,1]'
    n = p1.shape[1]

    # A matrix
    A = np.zeros((2 * n, 9), dtype=int)

    for i in range(n):
        x = p1[0, i]
        y = p1[1, i]
        u = p2[0, i]
        v = p2[1, i]


        A[(i * 2) + 1, :] = [-u, -v, -1, 0, 0, 0, x * u, x * v, x]
        # print(A)

        A[(i * 2), :] = [0, 0, 0, -u, -v, -1, y * u, y * v, y]
        # print(A)

    _, _, v = np.linalg.svd(A)

    H2to1 = (v[-1, :] / v[-1, -1]).reshape(3, 3)

    return H2to1


def ransacH(matches, locs1, locs2, num_iter=5000, tol=2):
    '''
    Returns the best homography by computing the best set of matches using
    RANSAC
    INPUTS
        locs1 and locs2 - matrices specifying point locations in each of the images
        matches - matrix specifying matches between these two sets of point locations
        nIter - number of iterations to run RANSAC
        tol - tolerance value for considering a point to be an inlier

    OUTPUTS
        bestH - homography matrix with the most inliers found during RANSAC
    '''
    ###########################
    num_kps = matches.shape[0]  # the number of detected and matched key points

    im1_orig_kps = np.ones((3, num_kps))
    im1_orig_kps[0:2, :] = np.transpose(locs1[matches[:, 0], 0:2])  # matched points in image1 for prediction

    im2_orig_kps = np.ones((3, num_kps))
    im2_orig_kps[0:2, :] = np.transpose(locs2[matches[:, 1], 0:2])  # matched points in image1 for prediction

    inline_all_H = np.zeros((num_iter, 1))
    all_H = np.zeros((num_iter, 3, 3))


    for i in range(num_iter):
        print("the" + str(i) + "th iteration")

        # generate 4 random pairs
        rand_kp_ind = np.random.randint(0, num_kps, 4)

        # the x,y coordinate of chosen point
        im1_chosen_kps = im1_orig_kps[:, rand_kp_ind]
        im2_chosen_kps = im2_orig_kps[:, rand_kp_ind]

        # compute h based on the randomly chosen points
        H = computeH(im1_chosen_kps[0:2, :], im2_chosen_kps[0:2, :])
        all_H[i, :, :] = H

        # predicted the points in i, image1 from image2 based on H
        im1_pred_kps = np.matmul(H, im2_orig_kps)
        # Normalize the x,y coordinate according to z
        normal = np.matlib.repmat(im1_pred_kps[2, :], 2, 1)
        im1_pred_kps = np.divide(im1_pred_kps[0:2, :], normal)

        inline = 0

        # for every predicted point, compute the distance between it and the original points
        for j in range(4):
            dist = np.linalg.norm(im1_pred_kps[:, j] - im1_orig_kps[0:2, j])
            if dist < tol:
                inline += 1

        inline_all_H[i, 0] = inline

    bestH = all_H[np.argmax(inline_all_H), :, :]

    return bestH


if __name__ == '__main__':
    compareX, compareY = makeTestPattern()
    im1 = cv2.imread('../data/incline_L.png')
    im2 = cv2.imread('../data/incline_R.png')
    locs1, desc1 = briefLite(im1, compareX, compareY)
    locs2, desc2 = briefLite(im2, compareX, compareY)
    matches = briefMatch(desc1, desc2)
    plotMatches(im1, im2, matches, locs1, locs2)
    bestH = ransacH(matches, locs1, locs2, num_iter=5000, tol=2)
    np.save('../results/q6_1', bestH)
    print(1)

