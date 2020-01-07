import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches
from LucasKanadeBasis import LucasKanadeBasis
from LucasKanade import LucasKanade

# write your script here, we recommend the above libraries for making your animation
if __name__ == '__main__':

    rect_Tem = np.array([101, 61, 155, 107]).astype(np.double)
    rect_Ori = rect_Tem.copy()
    rect_w = rect_Tem[2] - rect_Tem[0]
    rect_h = rect_Tem[3] - rect_Tem[1]

    frame = np.load('../data/sylvseq.npy')
    numFrame = frame.shape[2]
    bases = np.load('../data/sylvbases.npy')

    sylvseqrects = np.zeros((numFrame-1, 4))

    for i in range(numFrame - 1):
        prev_fra = frame[:, :, i]
        curr_fra = frame[:, :, i+1]

        p_LK_Tem = LucasKanadeBasis(prev_fra, curr_fra, rect_Tem, bases)
        p_LK_Ori = LucasKanade(prev_fra, curr_fra, rect_Ori)

        rect_Tem[0] += p_LK_Tem[1]
        rect_Tem[1] += p_LK_Tem[0]
        rect_Tem[2] += p_LK_Tem[1]
        rect_Tem[3] += p_LK_Tem[0]

        rect_Ori[0] += p_LK_Ori[1]
        rect_Ori[1] += p_LK_Ori[0]
        rect_Ori[2] += p_LK_Ori[1]
        rect_Ori[3] += p_LK_Ori[0]

        sylvseqrects[i, :] = rect_Tem
        print("this is the " + str(i) +"th image" + str(rect_Tem))

        if i == 0 or i == 199 or i == 299 or i == 349 or i == 399:
            plt.figure()
            currentAxis = plt.gca()

            rect_draw_Ori = patches.Rectangle((rect_Ori[0], rect_Ori[1]), rect_w, rect_h, linewidth=1, edgecolor='b', facecolor='none')
            currentAxis.add_patch(rect_draw_Ori)

            rect_draw_Tem = patches.Rectangle((rect_Tem[0], rect_Tem[1]), rect_w, rect_h, linewidth=1, edgecolor='r',
                                              facecolor='none')
            currentAxis.add_patch(rect_draw_Tem)

            plt.imshow(curr_fra, cmap='gray')
            plt.savefig("../data/sylvseqrects" + str(i+1) + ".png")
            plt.show()

        np.save('../data/sylvseqrects.npy', sylvseqrects)

    carseqrects = np.load('../data/sylvseqrects.npy')
    print(sylvseqrects)



