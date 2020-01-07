import numpy as np
from submission import *
from helper import *

'''
Q3.3:
    1. Load point correspondences
    2. Obtain the correct M2
    3. Save the correct M2, C2, and P to q3_3.npz
'''


def test_M2_solution(pts1, pts2, intrinsics):
	'''
	Estimate all possible M2 and return the correct M2 and 3D points P
	:param pred_pts1:
	:param pred_pts2:
	:param intrinsics:
	:return: M2, the extrinsics of camera 2
			 C2, the 3x4 camera matrix
			 P, 3D points after triangulation (Nx3)
	'''
	K1 = intrinsics['K1']
	K2 = intrinsics['K2']

	im1 = plt.imread('../data/im1.png')

	M = max(im1.shape)

	F = eightpoint(pts1, pts2, M)
	E = essentialMatrix(F, K1, K2)

	M1 = np.array([[1.0, 0, 0, 0],
				   [0, 1.0, 0, 0],
				   [0, 0, 1.0, 0]])
	M2s = camera2(E)

	C1 = np.dot(K1, M1)

	min_error = 1e12
	min_index = 0
	C2 = 0
	P = 0

	for i in range(4):
		C2 = np.dot(K2, M2s[:, :, i])
		P, error = triangulate(C1, pts1, C2, pts2)
		if error < min_error:
			min_error = np.copy(error)
			min_index = np.copy(i)
			C2 = np.copy(C2)
			P = np.copy(P)

	M2 = M2s[:, :, min_index]
	print("find M2 error: ", min_error)

	return M2, C2, P


if __name__ == '__main__':
	data = np.load('../data/some_corresp.npz')
	pts1 = data['pts1']
	pts2 = data['pts2']
	intrinsics = np.load('../data/intrinsics.npz')

	M2, C2, P = test_M2_solution(pts1, pts2, intrinsics)
	np.savez('q3_3', M2=M2, C2=C2, P=P)
