import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.image as img
import matplotlib.patches as patches
from LucasKanade import LucasKanade


# write your script here, we recommend the above libraries for making your animation


fig, ax = plt.subplots(1)
save_frame = np.array([1, 100, 200, 300])

if __name__ == '__main__':

    frames = np.load('../data/carseq.npy')
    numFrames = frames.shape[2]
    w_image = frames.shape[1]
    h_image = frames.shape[0]
    rect = np.array([59, 116, 145, 151])
    w = rect[2] - rect[0]
    h = rect[3] - rect[1]

    carseqrects = np.zeros((numFrames-1, 4))
    for i in range(numFrames - 1):
        previous_frame = frames[:, :, i]
        current_frame = frames[:, :, i + 1]
        move_pattern = LucasKanade(previous_frame, current_frame, rect)
        move_y, move_x = move_pattern

        rect = rect.astype(np.double)
        rect[0] += move_x
        rect[1] += move_y
        rect[2] += move_x
        rect[3] += move_y

        print("the ith frame" + str(i)+str(rect))


        if i == 0 or i == 99 or i == 199 or i == 299 or i==399:
            plt.figure()
            currentAxis = plt.gca()
            rect_draw = patches.Rectangle((rect[0], rect[1]), w, h, linewidth=1, edgecolor='r', facecolor='none')
            currentAxis.add_patch(rect_draw)
            plt.imshow(current_frame, cmap='gray')
            plt.savefig("../data/carseqrects"+str(i+1)+".png")
            # plt.show()
            print("this is" + str(rect))

        carseqrects[i, :] = rect.reshape(1, 4)

    np.save('../data/carseqrects.npy', carseqrects)

    carseqrects = np.load('../data/carseqrects.npy')
    print(carseqrects)








