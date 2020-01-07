import numpy as np
import cv2


def createGaussianPyramid(im, sigma0=1,
                          k=np.sqrt(2), levels=[-1, 0, 1, 2, 3, 4]):
    """
    create a gaussian fiilter pyramid
    :param im: image with 3 color channel
    :param sigma0: initial standard deviation of the gaussian filter
    :param k:
    :param levels: the exponent index of standard deviation in different layer
    :return: image pyramid of shape[imH, imW, len(levels)]
    """
    # convert the colorful picture to gray and normalize it to 0-1
    if len(im.shape) == 3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    if im.max() > 10:
        im = np.float32(im) / 255

    im_pyramid = []
    # conv the image with gaussian filter in different scale and stack the result as a pyramid
    for i in levels:
        sigma_ = sigma0 * k ** i
        im_pyramid.append(cv2.GaussianBlur(im, (0, 0), sigma_))
    im_pyramid = np.stack(im_pyramid, axis=-1)
    return im_pyramid


def displayPyramid(im_pyramid):
    im_pyramid = np.split(im_pyramid, im_pyramid.shape[2], axis=2)
    im_pyramid = np.concatenate(im_pyramid, axis=1)
    im_pyramid = cv2.normalize(im_pyramid, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    cv2.imshow('Pyramid of image', im_pyramid)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def createDoGPyramid(gaussian_pyramid, levels=[-1, 0, 1, 2, 3, 4]):
    """
    Produces DoG Pyramid
    INPUTS
        gaussian_pyramid - A matrix of grayscale images of size
                            [imH, imW, len(levels)]
        levels           - the levels of the pyramid where the blur at each level is
                            outputs

    OUTPUTS
        DoG_pyramid - size (imH, imW, len(levels) - 1) matrix of the DoG pyramid
                   created by differencing the Gaussian Pyramid input
        DoG_levels  - all but the very first item from the levels vector
    """

    DoG_pyramid = []
    ################
    len_DoG = len(levels) - 1
    for i in range(len_DoG):
        DoG_pyramid.append(gaussian_pyramid[:, :, i + 1] - gaussian_pyramid[:, :, i])
    DoG_pyramid = np.stack(DoG_pyramid, axis=-1)

    DoG_levels = levels[1:]
    return DoG_pyramid, DoG_levels


def computePrincipalCurvature(DoG_pyramid):
    """
    Takes in DoGPyramid generated in createDoGPyramid and returns
    PrincipalCurvature,a matrix of the same size where each point contains the
    curvature ratio R for the corre-sponding point in the DoG pyramid

    INPUTS
        DoG_pyramid - size (imH, imW, len(levels) - 1) matrix of the DoG pyramid

    OUTPUTS
        principal_curvature - size (imH, imW, len(levels) - 1) matrix where each
                          point contains the curvature ratio R for the
                          corresponding point in the DoG pyramid
    """
    principal_curvature = None

    ##################
    # calculate the element of hessian matrix
    sobelxx = cv2.Sobel(DoG_pyramid, cv2.CV_64F, 2, 0, ksize=3)
    sobelyy = cv2.Sobel(DoG_pyramid, cv2.CV_64F, 0, 2, ksize=3)
    sobelxy = cv2.Sobel(DoG_pyramid, cv2.CV_64F, 1, 1, ksize=3)

    # calculate the trace and determinant of hessian matrix
    Tr = sobelxx + sobelyy
    Det = np.multiply(sobelxx, sobelyy) - np.square(sobelxy)

    # calculate the principal curvature value with Trace and Determinant
    principal_curvature = np.divide(np.square(Tr), Det)

    return principal_curvature


def getLocalExtrema(DoG_pyramid, DoG_levels, principal_curvature,
                    th_contrast=0.03, th_r=12):
    '''
    Returns local extrema points in both scale and space using the DoGPyramid

    INPUTS
        DoG_pyramid - size (imH, imW, len(levels) - 1) matrix of the DoG pyramid
        DoG_levels  - The levels of the pyramid where the blur at each level is
                      outputs
        principal_curvature - size (imH, imW, len(levels) - 1) matrix contains the
                      curvature ratio R
        th_contrast - remove any point that is a local extremum but does not have a
                      DoG response magnitude above this threshold
        th_r        - remove any edge-like points that have too large a principal
                      curvature ratio
     OUTPUTS
        locsDoG - N x 3 matrix where the DoG pyramid achieves a local extrema in both
               scale and space, and also satisfies the two thresholds.
    '''
    locsDoG = None

    ##############

    # filter out the the pixel which is less than the th_contrast
    # filter out the plane
    Dog_mask = DoG_pyramid > th_contrast

    # filter out the pixel which is greater than the th_r
    # filter out the edge
    Pcur_mask = principal_curvature < th_r

    # remain the point that satisfied the above two condition
    final_mask = np.bitwise_and(Dog_mask, Pcur_mask)

    # return the coordinate of the qualified points in different scale
    coordinates_levels = np.where(final_mask == True)
    coordinates_levels = np.stack(coordinates_levels, axis=-1)

    # non maximum suppression
    # if the key point is greater than the 8+2 = 10 pixels "around" it, keep it, other wise, delete it
    kept_keypoint = []
    for the_th_keypoint in range(coordinates_levels.shape[0]):
        # x, y, z coordinate for the keypoint in the DoG pyramid
        x = coordinates_levels[the_th_keypoint][0]
        y = coordinates_levels[the_th_keypoint][1]
        z = coordinates_levels[the_th_keypoint][2]

        comparison_list = []
        x_left = x - 1
        x_right = x + 2
        y_top = y - 1
        y_bottom = y + 2
        z_front = z - 1
        z_back = z + 2

        # the leftest and rightest pixel only have 18 nearby pixels to compare
        if x == 0:
            x_left = x
        elif x == DoG_pyramid.shape[0] - 1:
            x_right = x + 1

        # the top and bottom pixel only have 18 nearby pixels to compare
        if y == 0:
            y_top = y
        elif y == DoG_pyramid.shape[1] - 1:
            y_bottom = y + 1

        # the last and the first layer of the pyramid only have two layer to compare
        if z == 0:  # the first layer of the pyramid
            z_front = z
        elif z == DoG_pyramid.shape[2] - 1:  # the last layer of the pyramid
            z_back = z + 1

        # compare the current pixel withit nearny 3*3 box
        keep_this_point_or_not = True
        """
        for x_i in range(x_left, x_right):
            for y_i in range(y_top, y_bottom):
                for z_i in range(z_front, z_back):
                    center_pixel = DoG_pyramid[x, y, z]
                    nearby_pixel = DoG_pyramid[x_i, y_i, z_i]
                    if center_pixel < nearby_pixel:
                        keep_this_point_or_not = False
                        break  # jump out of the judgement loop if the point is not the max point
                    else:
                        continue
                else:
                    continue  # jump out of the judgement loop if the point is not the max point
                break
            else:
                continue  # jump out of the judgement loop if the point is not the max point
            break
        """
        for x_i in range(x_left, x_right):
            for y_i in range(y_top, y_bottom):
                for z_i in range(z_front, z_back):
                    center_pixel = DoG_pyramid[x, y, z]
                    nearby_pixel = DoG_pyramid[x_i, y_i, z_i]
                    if center_pixel < nearby_pixel:
                        keep_this_point_or_not = False
                        break  # jump out of the judgement loop if the point is not the max point
                    else:
                        continue
                else:
                    continue  # jump out of the judgement loop if the point is not the max point
                break
            else:
                continue  # jump out of the judgement loop if the point is not the max point
            break

        if keep_this_point_or_not == True:
            kept_keypoint.append(np.array([x, y, z]))

    # combine all the interested point into an array
    kept_keypoint = np.array(kept_keypoint).astype(np.float)
    locsDoG = kept_keypoint.copy()
    locsDoG[:, 1] = kept_keypoint[:, 0]
    locsDoG[:, 0] = kept_keypoint[:, 1]

    return locsDoG


def DoGdetector(im, sigma0=1, k=np.sqrt(2), levels=[-1, 0, 1, 2, 3, 4],
                th_contrast=0.03, th_r=12):
    '''
    Putting it all together

    INPUTS          Description
    --------------------------------------------------------------------------
    im              Grayscale image with range [0,1].

    sigma0          Scale of the 0th image pyramid.

    k               Pyramid Factor.  Suggest sqrt(2).

    levels          Levels of pyramid to construct. Suggest -1:4.

    th_contrast     DoG contrast threshold.  Suggest 0.03.

    th_r            Principal Ratio threshold.  Suggest 12.


    OUTPUTS         Description
    --------------------------------------------------------------------------

    locsDoG         N x 3 matrix where the DoG pyramid achieves a local extrema
                    in both scale and space, and satisfies the two thresholds.

    gauss_pyramid   A matrix of grayscale images of size (imH,imW,len(levels))
    '''

    ##########################

    gauss_pyramid = createGaussianPyramid(im)
    # kpDet.displayPyramid(gaussian_pyramid)

    # 1.2 create the DoG pyramid based on the gaussian pyramid
    DoG_pyramid, DoG_levels = createDoGPyramid(gauss_pyramid)
    # kpDet.displayPyramid(DoG_pyramid)

    # 1.3 calculate the principal curvature of every point
    principal_curvature = computePrincipalCurvature(DoG_pyramid)

    # 1.4 Detecting Extrema
    locsDoG = getLocalExtrema(DoG_pyramid, DoG_levels, principal_curvature, th_contrast, th_r)


    return locsDoG, gauss_pyramid


if __name__ == '__main__':
    # test gaussian pyramid
    levels = [-1, 0, 1, 2, 3, 4]
    im_ori = cv2.imread('../data/incline_L.png')
    im_gray = cv2.cvtColor(im_ori, cv2.COLOR_BGR2GRAY)
    im_norm_gray = im_gray/255

    im_pyr = createGaussianPyramid(im_norm_gray)
    DoG_pyr, DoG_levels = createDoGPyramid(im_pyr, levels)
    displayPyramid(DoG_pyr)

    # test DoG detector
    locsDoG, gaussian_pyramid = DoGdetector(im_norm_gray)

    # display the corner
    # """
    for i in range(locsDoG.shape[0]):
       cv2.circle(img=im_ori, center=(int(locsDoG[i, 0]), int(locsDoG[i, 1])), radius=1, color=(255, 0, 255), thickness=-1)

    cv2.imshow('image', im_ori)
    cv2.waitKey(0)  # press any key to exit
    # """

