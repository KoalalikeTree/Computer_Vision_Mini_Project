"""""""""""""""""""""""""""""EVALUATION"""""""""""""""""""""""""""""""""""

import os
import matplotlib
import torch
import numpy
from matplotlib import pyplot as plt
from skimage.filters import threshold_otsu
from skimage.color import rgb2gray
from skimage.segmentation import clear_border
from skimage.measure import label
import skimage.morphology as morphology
from skimage.measure import regionprops
import skimage.io
from q4 import findLetters
import numpy as np
import torch.nn as nn

# do not include any more libraries here!
# no opencv, no sklearn, etc!
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)



# detect the box of the image in the folder

class lenet5(nn.Module):
    def __init__(self):
        super(lenet5, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(3, 3)),
                                   nn.ReLU(),
                                   nn.MaxPool2d(kernel_size=(2, 2), stride=2),
                                   nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(3, 3)),
                                   nn.ReLU(),
                                   nn.MaxPool2d(kernel_size=(2, 2), stride=2))
        self.conv2 = nn.Sequential(nn.Conv2d(16, 120, kernel_size=(5, 5)),
                                   nn.ReLU())
        self.fc1 = nn.Sequential(nn.Linear(120, 84),
                                 nn.ReLU(),
                                 nn.Linear(84, 47),
                                 nn.LogSoftmax(dim=-1))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(-1, 120)
        x = self.fc1(x)
        return x

def get_letter_image():
    img_number = 0
    images = {}
    char_rows = {}

    for img_num, img in enumerate(os.listdir('../images')):
        im1 = skimage.img_as_float(skimage.io.imread(os.path.join('../images', img)))
        bboxes, bw = findLetters(im1)

        # plt.imshow(bw, cmap = 'gray')
        for bbox in bboxes:
            minr, minc, maxr, maxc = bbox
            rect = matplotlib.patches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                                fill=False, edgecolor='#FF79A0', linewidth=0.5)
            plt.gca().add_patch(rect)

        img_number += 1
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
            y_center = (y_bottom + y_top) / 2
            x_center = (x_left + x_right) / 2

            # if the vertical distance between the current box with the last one is big
            # it is another row
            if (y_center - previous_char_y_center) >= (y_bottom - y_top):
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

        letter_vectors = np.empty((num_char, 1, 32, 32))
        for i in range(num_char):
            y_top2, x_left2, y_bottom2, x_right2, row_id = sorted_bboxes[i]
            y_center = (y_bottom2 + y_top2) / 2
            x_center = (x_left2 + x_right2) / 2

            letter = np.array(bw[y_top2:y_bottom2 + 1, x_left2:x_right2 + 1], dtype=np.int)
            letter = morphology.binary_erosion(letter, np.ones((5, 5)))
            padding_h = np.ones((letter.shape[0], 10))
            padding_v = np.ones((20, letter.shape[1] + 20))
            letter = np.hstack((padding_h, letter, padding_h))
            letter = np.vstack((padding_v, letter, padding_v))

            letter = skimage.transform.resize(letter, (28, 28), preserve_range=True, anti_aliasing=False)
            letter = letter.transpose()
            letter =  1 - letter
            letter_vectors[i, 0, :, :] = letter
            # plt.figure(1)
            # plt.imshow(letter_vectors[i, 0, :, :])
            # plt.show()

        images['img'+str(img_num)] = letter_vectors
        char_rows['img'+str(img_num)] = char_row

    return images, char_rows


##########################
##### load the weight#####
##########################

letter_dict = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
         10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'I', 19: 'J',
         20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'O', 25: 'P', 26: 'Q', 27: 'R', 28: 'S', 29: 'T',
         30: 'U', 31: 'V', 32: 'W', 33: 'X', 34: 'Y', 35: 'Z', 36: 'a', 37: 'b', 38: 'd', 39: 'e',
         40: 'f', 41: 'g', 42: 'h', 43: 'n', 44: 'q', 45: 'r', 46: 't'}
images, char_rows= get_letter_image()

model = lenet5().float()
trained_model = torch.load('q7_1_4_model_parameter.pkl', map_location='cpu')
model.load_state_dict(trained_model)

import torchvision
import torchvision.transforms as transforms
batch_size = 64

transforms = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize([0.5],[0.5])])

torchvision.datasets.EMNIST.url = 'http://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/gzip.zip'

testset = torchvision.datasets.EMNIST('./data', split='balanced', train=False,
                                     download=True, transform= transforms)
test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                          shuffle=False, num_workers=0)

test_correct = 0
for data in test_loader:
    # get the inputs
    inputs_test = torch.autograd.Variable(data[0])
    labels = torch.autograd.Variable(data[1])

    # get output
    y_pred_test = model(inputs_test)
    loss = nn.functional.cross_entropy(y_pred_test, labels)

    predicted = torch.max(y_pred_test, 1)[1]
    test_correct += torch.sum(predicted == labels.data).item()

    inputs_visualization = inputs_test.numpy()
    labels_visualization = predicted.numpy()
    # for i in range(inputs_visualization.shape[0]):
    #     plt.figure(1)
    #     plt.imshow(inputs_visualization[i, 0, :, :].transpose())
    #     plt.title(str(labels_visualization[i]) + letter_dict[labels_visualization[i]])
    #     plt.show()
    # print(1)
test_acc = test_correct/len(testset)

print('Test accuracy: {}'.format(test_acc))

for i in range(len(images)):
    letter_vectors = images['img'+str(i)]
    letter_vectors -= 0.5
    letter_vectors /= 0.5

    num_char = letter_vectors.shape[0]
    char_row = char_rows['img'+str(i)]

    letter_vectors = torch.from_numpy(letter_vectors).float()
    inputs = torch.autograd.Variable(letter_vectors)

    prediction = model(inputs)
    predicted = torch.max(prediction, 1)[1]

    predicted = predicted.numpy()

    # predict the letter based on model
    for i in range(letter_vectors.shape[0]):
        plt.figure(1)
        plt.imshow(letter_vectors[i, 0, :, :])
        plt.title(str(predicted[i]) + letter_dict[predicted[i]])
        plt.show()
    print(1)


    result_setence = ''
    for i in range(num_char):
        # find the corresponding letter in the dictionary
        # add it to the scentence
        result_setence += letter_dict[predicted[i]]

        # if t
        if i < num_char-1:
            if char_row[i] != char_row[i+1]:
                result_setence += ' / '

    print(result_setence)