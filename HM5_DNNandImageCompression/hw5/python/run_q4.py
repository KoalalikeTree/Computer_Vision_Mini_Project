import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.io
import skimage.transform
import skimage.filters
import skimage.morphology
import skimage.segmentation

from nn import *
from q4 import *
# do not include any more libraries here!
# no opencv, no sklearn, etc!
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)


img_number = 0

for img in os.listdir('../images'):
    im1 = skimage.img_as_float(skimage.io.imread(os.path.join('../images',img)))
    bboxes, bw = findLetters(im1)

    # plt.imshow(bw, cmap = 'gray')
    for bbox in bboxes:
        minr, minc, maxr, maxc = bbox
        rect = matplotlib.patches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                fill=False, edgecolor='#FF79A0', linewidth=0.5)
        plt.gca().add_patch(rect)

    img_number += 1
    filename = 'box' + str(img_number)
    plt.savefig(filename)
    # plt.show()

    sorted_bboxes = []
    # find the rows using..RANSAC, counting, clustering, etc.
    ##########################
    ##### your code here #####
    ##########################
    num_char = len(bboxes)
    previous_char_y_center = 0
    row_id = 0
    char_row = np.zeros((num_char, 1))
    for i in range(num_char):
        y_top, x_left, y_bottom, x_right = bboxes[i]
        y_center = (y_bottom + y_top)/2
        x_center = (x_left + x_right)/2

        # if the vertical distance between the current box with the last one is big
        # it is another row
        if(y_center-previous_char_y_center) >= (y_bottom-y_top):
            row_id += 1

        previous_char_y_center = y_center
        char_row[i] = row_id
        sorted_bboxes.append((y_top, x_left, y_bottom, x_right, row_id))
    # at first, sort the boxes according to the row number
    # second, sort the box from left to right
    sorted_bboxes = sorted(sorted_bboxes, key=lambda x: (x[4], x[3]))

    # crop the bounding boxes
    # note.. before you flatten, transpose the image (that's how the dataset is!)
    # consider doing a square crop, and even using np.pad() to get your images looking more like the dataset
    ##########################
    ##### your code here #####
    ##########################

    letter_vectors = np.empty((0, 1024))
    for i in range(num_char):
        y_top2, x_left2, y_bottom2, x_right2, row_id = sorted_bboxes[i]
        y_center = (y_bottom2 + y_top2) / 2
        x_center = (x_left2 + x_right2) / 2

        letter = np.array(bw[y_top2:y_bottom2+1, x_left2:x_right2+1], dtype=np.int)
        letter = skimage.morphology.binary_erosion(letter, np.ones((5, 5)))
        padding_h = np.ones((letter.shape[0], 20))
        padding_v = np.ones((20, letter.shape[1]+40))
        letter = np.hstack((padding_h, letter, padding_h))
        letter = np.vstack((padding_v, letter, padding_v))

        # plt.imshow(letter, cmap='gray')
        # plt.show()

        # resize the image to the dimension of the nn
        letter = skimage.transform.resize(letter, (32, 32), preserve_range=True, anti_aliasing=False)

        # plt.imshow(letter, cmap='gray')
        # plt.show()
        # plt.imshow(letter.transpose(), cmap='gray')
        # plt.show()

        # The image in the nn is transposed
        letter = np.transpose(letter).reshape((1, 1024))


        letter_vectors = np.append(letter_vectors, letter, axis=0)

    # load the weights
    # run the crops through your neural network and print them out
    import pickle
    import string
    letters = np.array([_ for _ in string.ascii_uppercase[:26]] + [str(_) for _ in range(10)])
    params = pickle.load(open('q3_weights.pickle', 'rb'))
    ##########################
    ##### your code here #####
    ##########################

    with open('q3_weights.pickle', 'rb') as handle:
        saved_params = pickle.load(handle)

    # implement the nn
    layer1 = forward(letter_vectors, params, 'layer1')
    probs = forward(layer1, params, 'output', softmax)

    predict = np.argmax(probs, axis=1)

    letter_result = np.copy(predict)

    result_setence = ''
    for i in range(num_char):
        # find the corresponding letter in the dictionary
        # add it to the scentence
        result_setence += letters[letter_result[i]]

        # if t
        if i < num_char-1:
            if char_row[i] != char_row[i+1]:
                result_setence += ' / '

    print(result_setence)