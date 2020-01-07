import cv2
import numpy as np
from numpy import matlib
from scipy.ndimage.morphology import distance_transform_edt
from planarH import ransacH
from BRIEF import briefLite,briefMatch,plotMatches


def imageStitching(im1, im2, H2to1):
    '''
    Returns a panorama of im1 and im2 using the given homography matrix. 
    Warps img2 into img1 reference frame using the provided warpH() function

    INPUT
        im1 and im2 - two images for stitching
        H2to1 - a 3 x 3 matrix encoding the homography that best matches the linear
                 equation
    OUTPUT
        Blends img1 and warped img2 and outputs the panorama image
    '''
    
    #######################################
    # TO DO ...
    pano_im = cv2.warpPerspective(im1, H2to1, (im2.shape[1], im2.shape[0]))

    cv2.imshow('pano_im', pano_im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return pano_im


def imageStitching_noClip(im1, im2, H2to1):
    '''
    Returns a panorama of im1 and im2 using the given homography matrix without clipping. 
    Warps img2 into img1 reference frame using the provided warpH() function


    INPUT
        im1 and im2 - two images for stitching
        H2to1 - a 3 x 3 matrix encoding the homography that best matches the linear
                 equation
    OUTPUT
        Blends img1 and warped img2 and outputs the panorama image
    '''
    
    ######################################
    # TO DO ...
    # 1. find the corner of the two image
    corner_im1 = np.zeros((2, 4))  # [:, 0] is the left top/ [:, 1] right top/ [:, 2] right bottom/ [:, 3] left bottom
    corner_im2 = np.zeros((2, 4))
    im1_H = im1.shape[0] - 1
    im1_W = im1.shape[1] - 1
    im2_H = im2.shape[0] - 1
    im2_W = im2.shape[1] - 1

    corner_im2[:, 0] = np.array([0, 0]).T
    corner_im2[:, 1] = np.array([0, im2_H]).T
    corner_im2[:, 3] = np.array([im2_W, 0]).T
    corner_im2[:, 2] = np.array([im2_W, im2_H]).T

    # 2. warp the corner of im2 with H2to1
    corner_im2_temp = np.vstack((corner_im2, np.ones((1, 4))))
    corner_im2_warped_no_norm = np.matmul(H2to1, corner_im2_temp)
    L = np.matlib.repmat(corner_im2_warped_no_norm[2, :], 2, 1)
    corner_im2_warped = np.divide(corner_im2_warped_no_norm[0:2, :], L)

    # 3. use the new max/min corner to define the size of the new image
    maxcorner_im2_X = np.max(corner_im2_warped[0, :]).astype(int)
    maxcorner_im2_Y = np.max(corner_im2_warped[1, :]).astype(int)
    mincorner_im2_X = np.min(corner_im2_warped[0, :]).astype(int)
    mincorner_im2_Y = np.min(corner_im2_warped[1, :]).astype(int)
    new_size_W_right = np.maximum(maxcorner_im2_X, im1_W)
    new_size_W_left = np.minimum(mincorner_im2_X, 0)
    new_size_H_top = np.maximum(maxcorner_im2_Y, im1_H)
    new_size_H_down = np.minimum(mincorner_im2_Y, 0)
    new_size_W = new_size_W_right - new_size_W_left
    new_size_H = new_size_H_top - new_size_H_down

    # 4. warp image 1 with translation matrix to get rid of negative coordinate
    M = np.array([[1, 0, np.abs(new_size_W_left)],
                 [0, 1, np.abs(new_size_H_down)],
                 [0, 0, 1]],
                 dtype=np.float32)

    im1_warped = cv2.warpPerspective(im1, M, (new_size_W, new_size_H))

    # 5. warp image 2 with the translnp.matmul(M, H2to1)ation matrix * H2to1
    im2_M = np.matmul(M, H2to1)
    im2_warped = cv2.warpPerspective(im2, im2_M, (new_size_W, new_size_H))

    pano_im = np.maximum(im1_warped, im2_warped)

    return pano_im



def generatePanaroma(im1, im2):
    '''
    Generate and save panorama of im1 and im2.

    INPUT
        im1 and im2 - two images for stitching
    OUTPUT
        Blends img1 and warped img2 (with no clipping) 
        and saves the panorama image.
    '''

    ######################################
    # TO DO ...
    pano_im = imageStitching_noClip(im1, im2, H2to1)

    cv2.imshow('pano_im', pano_im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    im1 = cv2.imread('../data/incline_L.png')
    im2 = cv2.imread('../data/incline_R.png')
    H2to1 = np.load('../results/q6_1.npy')
    np.save('../results/q6_1.npy', H2to1)
    H2to1 = H2to1.reshape(3, 3)
    generatePanaroma(im1, im2)