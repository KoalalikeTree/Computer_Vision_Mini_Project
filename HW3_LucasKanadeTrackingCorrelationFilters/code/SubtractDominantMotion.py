import numpy as np
from LucasKanadeAffine import LucasKanadeAffine
from scipy import ndimage
from InverseCompositionAffine import InverseCompositionAffine

def SubtractDominantMotion(image1, image2):
	# Input:
	#	Images at time t and t+1 
	# Output:
	#	mask: [nxm]
    # put your implementation here
    
    mask = np.zeros(image1.shape, dtype=bool)

    M = InverseCompositionAffine(image1, image2)
    # M = LucasKanadeAffine(image1, image2)

    # to be continued
    affined_image1 = ndimage.affine_transform(image1, M, offset=0.0, output_shape=image2.shape)

    tramsform_filter = affined_image1 > 0
    tramsform_filter = ndimage.morphology.binary_erosion(tramsform_filter, structure=np.ones((10, 10)))

    image2_with_filter = image2 * tramsform_filter
    image1_affine_with_filter = affined_image1 * tramsform_filter
    difference = np.abs(image2_with_filter - image1_affine_with_filter)

    thre = 0.3
    mask[difference > thre] = 1

    mask = ndimage.morphology.binary_dilation(mask, structure=np.ones((7, 7)))

    return mask

