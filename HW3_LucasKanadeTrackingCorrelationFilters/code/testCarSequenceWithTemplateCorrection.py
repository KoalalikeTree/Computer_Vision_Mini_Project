import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches
from LucasKanade import LucasKanade

# write your script here, we recommend the above libraries for making your animation
if __name__ == '__main__':
    frames = np.load('../data/carseq.npy')
    numFrames = frames.shape[2]
    w_image = frames.shape[1]
    h_image = frames.shape[0]

    ini_frame = frames[:, :, 0]
    ini_rect = np.array([59, 116, 145, 151]).astype(np.double)
    ini_rect_w = ini_rect[2] - ini_rect[0]
    ini_rect_h = ini_rect[3] - ini_rect[1]

    # old algorithm
    rect = np.array([59, 116, 145, 151])
    w = rect[2] - rect[0]
    h = rect[3] - rect[1]
    # old algorithm

    epi = 5

    curr_rect = ini_rect.copy()

    rect_positions = np.zeros((numFrames - 1, 4))

    for i in range(numFrames - 1):
        previous_frame = frames[:, :, i]
        current_frame = frames[:, :, i + 1]

        move_from_pre_frame = LucasKanade(previous_frame, current_frame, curr_rect)
        move_from_pre_frame_y, move_from_pre_frame_x = move_from_pre_frame

        para_p = np.zeros(move_from_pre_frame.shape)
        para_p[0] = move_from_pre_frame_x + curr_rect[1] - ini_rect[1]
        para_p[1] = move_from_pre_frame_y + curr_rect[0] - ini_rect[0]
        rect = rect.astype(np.double)

        # old algorithm
        rect[0] += move_from_pre_frame_x
        rect[1] += move_from_pre_frame_y
        rect[2] += move_from_pre_frame_x
        rect[3] += move_from_pre_frame_y
        # old algorithm

        move_from_ini_frame = LucasKanade(ini_frame, current_frame, ini_rect, para_p)

        move_from_ini_frame_y, move_from_ini_frame_x = move_from_ini_frame

        diff = np.linalg.norm(para_p - move_from_ini_frame)
        if diff < epi:
            curr_rect[0] = move_from_ini_frame_x + ini_rect[0]
            curr_rect[1] = move_from_ini_frame_y + ini_rect[1]
            curr_rect[2] = move_from_ini_frame_x + ini_rect[2]
            curr_rect[3] = move_from_ini_frame_y + ini_rect[3]
        else:
            curr_rect[0] += move_from_pre_frame_x
            curr_rect[1] += move_from_pre_frame_y
            curr_rect[2] += move_from_pre_frame_x
            curr_rect[3] += move_from_pre_frame_y

        print(str(i) + str(curr_rect))


        if i == 0 or i == 99 or i == 199 or i == 299 or i == 399:
            plt.figure()
            currentAxis = plt.gca()
            rect_draw = patches.Rectangle((curr_rect[0], curr_rect[1]), ini_rect_w, ini_rect_h, linewidth=1, edgecolor='r', facecolor='none')

            # old algorithm
            rect_draw2 = patches.Rectangle((rect[0], rect[1]), w, h, linewidth=1, edgecolor='g', facecolor='none')
            currentAxis.add_patch(rect_draw2)
            # old algorithm

            currentAxis.add_patch(rect_draw)
            plt.imshow(current_frame, cmap='gray')
            plt.savefig("../data/carseqrects-wcrt" + str(i+1) + ".png")
            plt.show()

        rect_positions[i, :] = np.transpose(curr_rect.reshape(4, 1))
        print(rect_positions)

    np.save('../data/carseqrects-wcrt.npy', rect_positions)

    carseqrects_wcrt = np.load('../data/carseqrects-wcrt.npy')

    print(carseqrects_wcrt)