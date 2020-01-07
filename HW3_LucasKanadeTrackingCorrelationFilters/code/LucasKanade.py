import numpy as np
from scipy.interpolate import RectBivariateSpline


def LucasKanade(It, It1, rect, p0=np.zeros(2)):
    """
     Input:
        It: template image
        It1: Current image
        rect: Current position of the car
        (top left, bot right coordinates)
        p0: Initial movement vector [dp_x0, dp_y0]
     Output:
        p: movement vector [dp_x, dp_y]
    """

    It_interp = interpolateImage(It)
    It1_interp = interpolateImage(It1)

    thre = 0.005

    p0 = p0.reshape(2, 1)
    p = np.zeros(2).reshape(2, 1)
    p += p0
    delta_p = np.array([50, 50])

    x_mesh_flatten, y_mesh_flatten = makeMeshGrid(rect)

    while np.linalg.norm(delta_p) > thre:

        x_mesh_flatten_It1 = x_mesh_flatten + p[1]
        y_mesh_flatten_It1 = y_mesh_flatten + p[0]

        A = computeA(y_mesh_flatten_It1, x_mesh_flatten_It1, It1_interp)

        b = computeb(It_interp, It1_interp, y_mesh_flatten, x_mesh_flatten, y_mesh_flatten_It1, x_mesh_flatten_It1)
        delta_p = np.linalg.lstsq(A, b)[0]
        p = p.reshape(2, 1) + delta_p.reshape(2, 1)

    return p


def interpolateImage(image):
    """
    Interpolate a 2D gray image

    :param image: the gray image in size M*N
    :return: image_interp: RectBivariateSpline class after interpolation
             image_x, the ascending array of x coordinate
             image_y, the ascending array of y coordinate
    """
    # 1. Interpolate It and It1 (Start)
    # 1.1 Get the int ascending array of the width and height of the It1 and It
    image_h = image.shape[0]
    image_y = np.arange(0, image_h, 1)

    image_w = image.shape[1]
    image_x = np.arange(0, image_w, 1)

    # 1. Interpolate It and It1
    # 1.2 Implement scipy.interpolate.RectBivariateSpline
    image_interp = RectBivariateSpline(image_y, image_x, image)
    # 1. Interpolate It and It1 (End)

    return image_interp


def computeA(y_mesh_flatten_It1, x_mesh_flatten_It1, It1_interp):
    """

    :param p: the p in |AΔp - b|² in shape 2 * 1
    :param It1_interp: the RectBivariateSpline class of the image It1
    :return: the A in |AΔp - b|² in shape M(the total pixel number in rect) * 2
    """

    # 2.2.4 Interpolate the I
    x_gradient_It1 = It1_interp.ev(y_mesh_flatten_It1, x_mesh_flatten_It1, dx=1).reshape(-1, 1)
    y_gradient_It1 = It1_interp.ev(y_mesh_flatten_It1, x_mesh_flatten_It1, dy=1).reshape(-1, 1)

    # 2.2.5 Compute A
    # I am not sure why the x y is flipped, but it seems right by visualization
    A = np.stack((x_gradient_It1, y_gradient_It1), axis=1)
   # 2. Compute A in |AΔp - b|² (end)

    A = A.squeeze()

    """
    # Unit Test for Step 2：
    #    visualize the x and y gradient in A
    rect_It1 = rect_It1.reshape(4, )
    x_gradient_It1_row = It1_interp.ev(x_mesh_flatten_It1, y_mesh_flatten_It1, dx=1).reshape(1, -1)
    x_gradient_It1_row = x_gradient_It1_row.reshape((rect_It1[3] - rect_It1[1] + 1).astype(int),
                                                    (rect_It1[2] - rect_It1[0] + 1).astype(int))
    x_gradient_It1_row = scale(x_gradient_It1_row, 0, 255).astype(np.uint8)
    x_gradient_It1_col = It1_interp.ev(x_mesh_flatten_It1, y_mesh_flatten_It1, dy=1).reshape(1, -1)
    x_gradient_It1_col = x_gradient_It1_col.reshape((rect_It1[3] - rect_It1[1] + 1).astype(int),
                                                    (rect_It1[2] - rect_It1[0] + 1).astype(int))
    x_gradient_It1_col = scale(x_gradient_It1_col, 0, 255).astype(np.uint8)
    cv.imshow('1', x_gradient_It1_row)
    cv.imshow('2', x_gradient_It1_col)
    cv.waitKey(0)
    """

    return A


def computeb(It_interp, It1_interp, x_mesh_flatten, y_mesh_flatten, x_mesh_flatten_It1, y_mesh_flatten_It1):
    """
    Compute difference of the intensity of rect in It and rect_It1 in It_interp

    :param It_interp: the RectBivariateSpline class of It image
    :param It1_interp:  the RectBivariateSpline class of It1 image
    :param rect: 4 * 1 vector, the top left, the top right, the bottom left, the bottom right in It
    :param rect_It1: 4 * 1 vector, the top left, the top right, the bottom left, the bottom right in It1
    :return: M * 1, vector, M is the number of pixel in rect
    """
    # compute the intensity of the rect_It in It
    intensity_It = It_interp.ev(x_mesh_flatten, y_mesh_flatten, dx=0, dy=0).reshape(1, -1)

    # compute the intensity of the rec_It1 in It1
    intensity_It1 = It1_interp.ev(x_mesh_flatten_It1, y_mesh_flatten_It1, dx=0, dy=0).reshape(1, -1)

    b = intensity_It - intensity_It1

    """Unit tes
    b_norm = scale(b, 0, 255).astype(np.uint8).reshape((rect_It[3]-rect_It[1] + 1), (rect_It[2]-rect_It[0] + 1))
    cv.imshow('b', b_norm)
    cv.waitKey(0)
    t"""
    b = b.reshape(-1,)

    return b


def makeMeshGrid(rect):

    rect_x = np.arange(np.floor(rect[0]), np.floor(rect[2] + 1), 1)
    rect_x += rect[0] % 1
    rect_y = np.arange(np.floor(rect[1]), np.floor(rect[3] + 1), 1)
    rect_y += rect[1] % 1

    # 2.2.2 Get the meshgrid based on 2.2.1 array
    x_mesh, y_mesh = np.meshgrid(rect_x, rect_y)
    # print("x_mesh.shape" + str(x_mesh.shape))
    # print("y_mesh.shape" + str(y_mesh.shape))

    # 2.2.3 Flatten the mesh grid
    x_mesh_flatten = x_mesh.flatten()
    y_mesh_flatten = y_mesh.flatten()

    return x_mesh_flatten, y_mesh_flatten


def scale(X, x_min, x_max):
    nom = (X-X.min(axis=0))*(x_max-x_min)
    denom = X.max(axis=0) - X.min(axis=0)
    denom[denom==0] = 1
    return x_min + nom/denom


if __name__ == '__main__':

    """
        Unit test for Step 1: Shift the image by (0.5, 0.5) and visualize it interpolation
    """

    # Unit test for step 1 Interpolate It and It1 (Start)
    # myphoto = cv.imread('../testImage/myphoto.jpg')
    # myphoto = cv.cvtColor(myphoto, cv.COLOR_BGR2GRAY)
    # myphoto2 = cv.imread('../testImage/myphoto4.jpg')
    # myphoto2 = cv.cvtColor(myphoto2, cv.COLOR_BGR2GRAY)
    #
    #
    # rect = np.array([260, 100, 470, 350])

    # cv.rectangle(myphoto, (rect[0], rect[1]), (rect[2], rect[3]), color=(255, 255, 255), thickness=2)
    # cv.imshow('myphoto', myphoto)


    # p = LucasKanade(myphoto, myphoto2, rect)
    # print(p)
    # cv.waitKey(0)
    # Unit test for step 1 Interpolate It and It1 (End)





