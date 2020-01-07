import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.filters
import skimage.morphology
import skimage.segmentation

# skimage package method
import matplotlib.patches as mpatches
from skimage import data
from skimage.filters import threshold_otsu
from skimage.color import rgb2gray
from skimage.segmentation import clear_border
from skimage.measure import label
from skimage.morphology import closing, square
from skimage.measure import regionprops
from skimage.color import label2rgb



# takes a color image
# returns a list of bounding boxes and black_and_white image
def findLetters(image):
    bboxes = []
    bw = None
    # insert processing in here
    # one idea estimate noise -> denoise -> greyscale -> threshold -> morphology -> label -> skip small boxes 
    # this can be 10 to 15 lines of code using skimage functions


    # convert a rgb image to gray image
    gray_image = rgb2gray(image)

    # apply threshold
    thresh = threshold_otsu(image)*0.8
    bw = closing(gray_image < thresh, square(15))

    # remove artifacts connected to image border
    cleared = bw.copy()
    bw = clear_border(cleared)
    # plt.imshow(bw, cmap='gray')
    # plt.show()

    # label image regions
    label_image = label(cleared)
    # image_label_overlay = label2rgb(label_image, image=image)

    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
    # ax.imshow(label_image, cmap='gray')

    for region in regionprops(label_image):

        # skip small images
        if region.area < 200:
            continue

        bboxes.append(region.bbox)

        # # draw rectangle around segmented coins
        #
        # minr, minc, maxr, maxc = region.bbox
        # rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
        #                           fill=False, edgecolor='red', linewidth=2)
        # ax.add_patch(rect)
    # plt.show()

    return bboxes, np.invert(bw)