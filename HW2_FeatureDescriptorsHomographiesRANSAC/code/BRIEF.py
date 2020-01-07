import os
import numpy as np
import cv2
import skimage
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist
from keypointDetect import DoGdetector
import numpy as np
import matplotlib.pyplot as plt


def makeTestPattern(patch_width=9, nbits=256):
    '''
    Creates Test Pattern for BRIEF
    Run this routine for the given parameters patch_width = 9 and n = 256

    INPUTS
        patch_width - the width of the image patch (usually 9)
        nbits       - the number of tests n in the BRIEF descriptor

    OUTPUTS
        compareX and compareY - LINEAR indices into the patch_width x patch_width image 
                                patch and are each (nbits,) vectors. 
    '''

    #############################
    compareX = np.random.randint(0, patch_width ** 2, nbits)
    compareY = np.random.randint(0, patch_width ** 2, nbits)
    return compareX, compareY


# load test pattern for Brief
test_pattern_file = '../results/testPattern.npy'
if os.path.isfile(test_pattern_file):
    # load from file if exists
    compareX, compareY = np.load(test_pattern_file)
else:
    # produce and save patterns if not exist
    compareX, compareY = makeTestPattern()
    if not os.path.isdir('../results'):
        os.mkdir('../results')
    np.save(test_pattern_file, [compareX, compareY])


def computeBrief(im, gaussian_pyramid, locsDoG,  # k, levels,
                 compareX, compareY):
    '''
    Compute Brief feature
     INPUT
     locsDoG - locsDoG are the keypoint locations returned by the DoG
               detector.
     levels  - Gaussian scale levels that were given in Section1.
     compareX and compareY - linear indices into the
                             (patch_width x patch_width) image patch and are
                             each (nbits,) vectors.


     OUTPUT
     locs - an m x 3 vector, where the first two columns are the image
    		 coordinates of keypoints and the third column is the pyramid
            level of the keypoints.
     desc - an m x n bits matrix of stacked BRIEF descriptors. m is the number
            of valid descriptors in the image and will vary.
    '''
    ##############################
    # TO DO ...
    # compute locs, desc here
    patch_width = 9
    nbits = 256

    num_Kps = locsDoG.shape[0]
    locs = np.zeros(locsDoG.shape, dtype=int)
    desc = np.zeros((num_Kps, nbits))

    H, W, L = gaussian_pyramid.shape

    for i in range(num_Kps):
        centerX = int(locsDoG[i, 0])
        centerY = int(locsDoG[i, 1])
        centerZ = int(locsDoG[i, 2])

        # within edge boundary
        if centerX < 4 or centerX > W - 5:
            continue
        if centerY < 4 or centerY > H - 5:
            continue

        P = gaussian_pyramid[centerY - 4:centerY + 5, centerX - 4:centerX + 5, centerZ]
        P = P.reshape(patch_width ** 2)

        tempDesc = np.zeros(nbits)

        for j in range(nbits):
            if P[compareX[j]] < P[compareY[j]]:
                desc[i, j] = 1
            else:
                desc[i, j] = 0

        locs[i, :] = np.array([centerX, centerY, centerZ])

    locs = np.delete(locs, (0), axis=0)
    desc = np.delete(desc, (0), axis=0)

    return locs, desc


def briefLite(im, compareX, compareY):
    '''
    INPUTS
        im - gray image with values between 0 and 1

    OUTPUTS
        locs - an m x 3 vector, where the first two columns are the image coordinates 
            of keypoints and the third column is the pyramid level of the keypoints
        desc - an m x n bits matrix of stacked BRIEF descriptors. 
            m is the number of valid descriptors in the image and will vary
            n is the number of bits for the BRIEF descriptor
    '''

    ###################
    locsDoG, gaussian_pyramid = DoGdetector(im)
    locs, desc = computeBrief(im, gaussian_pyramid, locsDoG, compareX, compareY)

    return locs, desc


def briefMatch(desc1, desc2, ratio=0.8):
    '''
    performs the descriptor matching
    INPUTS
        desc1, desc2 - m1 x n and m2 x n matrix. m1 and m2 are the number of keypoints in image 1 and 2.
                                n is the number of bits in the brief
    OUTPUTS
        matches - p x 2 matrix. where the first column are indices
                                        into desc1 and the second column are indices into desc2
    '''

    D = cdist(np.float32(desc1), np.float32(desc2), metric='hamming')
    # find smallest distance
    ix2 = np.argmin(D, axis=1)
    d1 = D.min(1)
    # find second smallest distance
    d12 = np.partition(D, 2, axis=1)[:, 0:2]
    d2 = d12.max(1)
    r = d1 / (d2 + 1e-10)
    is_discr = r < ratio
    ix2 = ix2[is_discr]
    ix1 = np.arange(D.shape[0])[is_discr]

    matches = np.stack((ix1, ix2), axis=-1)
    return matches


def plotMatches(im1, im2, matches, locs1, locs2):
    fig = plt.figure()
    # draw two images side by side
    imH = max(im1.shape[0], im2.shape[0])
    im = np.zeros((imH, im1.shape[1] + im2.shape[1]), dtype='uint8')
    im[0:im1.shape[0], 0:im1.shape[1]] = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im[0:im2.shape[0], im1.shape[1]:] = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    plt.imshow(im, cmap='gray')
    for i in range(matches.shape[0]):
        pt1 = locs1[matches[i, 0], 0:2]
        pt2 = locs2[matches[i, 1], 0:2].copy()
        pt2[0] += im1.shape[1]
        x = np.asarray([pt1[0], pt2[0]])
        y = np.asarray([pt1[1], pt2[1]])
        plt.plot(x, y, 'r', linewidth=0.1)
        plt.plot(x, y, 'b', linewidth=0.3)
    plt.show()


def rotateimage(im, angle):
    """
    rotate the input image and input keypoints
    :param im: normalized gray image
    :return: im_rota: the rotated image (the same size as the original one)
            keypoint_rota: the rotated key points of n * 2
    """
    rows = im.shape[0]
    cols = im.shape[1]

    # get the rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)

    # rotate the original image
    imgRotation = cv2.warpAffine(im, rotation_matrix, (int(rows * 1.2), int(cols * 1.5)))

    return imgRotation


def rotate2Dcoordinate(keypoints, angle):
    """
    Calculate the corresponding rotated coordinate
    :param keypoints: n * 2 matrix, each row is a coordinate
    :param angle: rotation angle (counterclockwise)
    :return: n * 2 matrix, each row is rotated coordinate
    """
    x_col = keypoints[:, 0]
    y_col = keypoints[:, 1]
    x = x_col * np.cos(angle * np.pi / 180) + y_col * np.sin(angle * np.pi / 180)
    x = x.reshape(x.shape[0], 1)
    y = y_col * np.cos(angle * np.pi / 180) + x_col * np.sin(angle * np.pi / 180)
    y = y.reshape(y.shape[0], 1)
    rotated_keypoints = np.concatenate((x, y), axis=1)
    return rotated_keypoints


def rotateimageandkeypoints(img, compareX, compareY):
    """
    1. rotate the original image
    2. detect the connection between the original image and the rotated image
    3. compare the rotated coordinate with the rotated one
    4. plot the match accuracy and its relationship with rotation angle
    :param img_ori: the original colorful image
    :param compareX, compareY: index for chosen pair in the patch around the corner
    :return: 1 * 36 vector, contain the accuracy for different rotation angle
    """
    # the consequence of different rotation angle
    img_ori = img
    img_norm = cv2.cvtColor(im1_ori, cv2.COLOR_BGR2GRAY)
    img_norm = np.divide(img_norm, 255)

    euclidean_dist = np.zeros((1, 36))
    for i in range(36):
        rotate_angele = 10 * i
        # 1.rotate the original image
        rotate_ori_image = rotateimage(img_ori, rotate_angele)
        rotate_image = rotateimage(img_norm, rotate_angele)

        # 2. detect the connection between the original image and the rotated image
        locs1, desc1 = briefLite(img_norm, compareX, compareY)
        locs2, desc2 = briefLite(rotate_image, compareX, compareY)

        # 3. compare the rotated coordinate with the rotated one
        rotated_locs1 = rotate2Dcoordinate(locs1[:, 0:2], rotate_angele)

        # 4. plot the match accuracy and its relationship with rotation angle
        detected_kps = np.zeros((rotated_locs1.shape[0], 2))
        detected_kps[:locs2.shape[0], 0:2] = locs2[:, 0:2]
        euclidean_dist[0, i] = np.sum(detected_kps - rotated_locs1)

        matches = briefMatch(desc1, desc2)

        # plot the match consequence
        plotMatches(img_ori, rotate_ori_image, matches, locs1, locs2)

    return euclidean_dist

    # rotate_kps = rotateimage(im1_ori, rotate_)

    # detect_kps =
    # test for image rotation
    # cv2.imshow('imgrotation', rotate_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


if __name__ == '__main__':
    # test makeTestPattern
    compareX, compareY = makeTestPattern()

    # test matches
    im1_ori = cv2.imread('../data/incline_L.png')
    im1 = cv2.cvtColor(im1_ori, cv2.COLOR_BGR2GRAY)
    im1 = np.divide(im1, 255)

    im2_ori = cv2.imread('../data/incline_R.png')
    im2 = cv2.cvtColor(im2_ori, cv2.COLOR_BGR2GRAY)
    im2 = np.divide(im2, 255)

    locs1, desc1 = briefLite(im1, compareX, compareY)
    locs2, desc2 = briefLite(im2, compareX, compareY)
    matches = briefMatch(desc1, desc2)
    plotMatches(im1_ori,im2_ori,matches,locs1,locs2)

    """
    im1_ori = cv2.imread('../data/model_chickenbroth.jpg')

    err_diff_angle = rotateimageandkeypoints(im1_ori, compareX, compareY)

    # plot the error of different angle
    x = range(36)
    plt.figure(figsize=(8, 4))  # 创建绘图对象
    plt.plot(x, np.abs(err_diff_angle.T), "b", linewidth=1)  # 在当前绘图对象绘图（X轴，Y轴，蓝色虚线，线宽度）
    plt.xlabel("Angles 0 - 360 degree")  # X轴标签
    plt.ylabel("(x_hat-x)+(y_hat-y)")  # Y轴标签
    plt.title("Relationship between Prediction Error and Rotation Angle")  # 图标题
    plt.show()  # 显示图
    plt.savefig("line.jpg")  # 保存图

    # test briefLite
    """
    """
    im = cv2.imread('../data/model_chickenbroth.jpg')
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im_gray_norm = np.divide(im_gray, 255)
    

    im1_ori = cv2.imread('../data/pf_scan_scaled.jpg')
    im1 = cv2.cvtColor(im1_ori, cv2.COLOR_BGR2GRAY)
    im1 = np.divide(im1, 255)

    im2_ori = cv2.imread('../data/pf_stand.jpg')
    im2 = cv2.cvtColor(im2_ori, cv2.COLOR_BGR2GRAY)
    im2 = np.divide(im2, 255)

    locs1, desc1 = briefLite(im1, compareX, compareY)
    locs2, desc2 = briefLite(im2, compareX, compareY)

    matches = briefMatch(desc1, desc2)

    plotMatches(im1_ori, im2_ori, matches, locs1, locs2)
    print(1)
    """