import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches
from SubtractDominantMotion import SubtractDominantMotion

# write your script here, we recommend the above libraries for making your animation
if __name__ == '__main__':

    frames = np.load('../data/aerialseq.npy')
    numFrames = frames.shape[2]
    w = frames.shape[1]
    h = frames.shape[0]

    # mask_stack 240，320，4
    mask_stack = np.zeros((h, w, 4))
    j = 0
    for i in range(numFrames-1):
        pre_fra = frames[:, :, i]
        cur_fra = frames[:, :, i+1]

        mask = SubtractDominantMotion(pre_fra, cur_fra)

        image = np.copy(cur_fra)

        image = np.stack((image, image, image), axis=2)
        image[:, :, 1][mask == 1] = 1

        if i == 29 or i == 59 or i == 89 or i == 119:

            # mask 240,320
            mask_stack[:, :, j] = mask.astype(np.int)
            print(np.where(mask_stack[:, :, j] > 0))
            j += 1

            # plt.figure()
            # plt.gca()
            # plt.imshow(image)
            # plt.savefig("../data/Aerialseqrects" + str(i + 1) + ".png")
            # plt.show()

    np.save("../code/aerialseqmasks.npy", mask_stack)


        # plt.figure()
        # plt.gca()
        # plt.imshow(image)
        # plt.show()
