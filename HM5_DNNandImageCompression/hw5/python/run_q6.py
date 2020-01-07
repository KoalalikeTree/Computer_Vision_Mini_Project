import numpy as np
import scipy.io
from nn import *
import matplotlib.pyplot as plt
from skimage.measure import compare_psnr as psnr

train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')

# we don't need labels now!
train_x = train_data['train_data']
valid_x = valid_data['valid_data']


dim = 32
# do PCA
##########################
##### your code here #####
##########################

n = train_x.shape[0]

means = np.mean(train_x, axis=0)

train_x_std = np.dot((train_x-means).T,(train_x-means))

train_x_std = np.divide(train_x_std, (n-1))

U, S, VT = np.linalg.svd(train_x_std)

projection_matrix = U[:, 0:dim]

# rebuild a low-rank version
lrank = np.dot(valid_x, projection_matrix)
##########################
##### your code here #####
##########################

# rebuild it
recon_valid = np.dot(lrank, projection_matrix.T)
##########################
##### your code here #####
##########################

chosen_class = np.random.choice(np.arange(36), 5, False) * 100

for num_plot, num_class in enumerate(chosen_class):
    plt.subplot(5, 2, 2*num_plot+1)
    plt.imshow(valid_x[num_class].reshape(32, 32).T, cmap='gray')
    plt.savefig('pca'+str(num_class)+'.png')
    plt.subplot(5, 2, 2*num_plot+2)
    plt.imshow(recon_valid[num_class].reshape(32, 32).T, cmap='gray')
    plt.savefig('pca'+str(num_class)+'.png')
plt.show()

# build valid dataset
##########################
##### your code here #####
##########################

# visualize the comparison and compute PSNR
##########################
##### your code here #####
##########################
num_valid_samples = valid_x.shape[0]
psnr_total = 0

for num_sample in range(num_valid_samples):
    psnr_total += psnr(valid_x[num_sample], recon_valid[num_sample])
psnr_average = psnr_total / num_valid_samples

print("Average psnr", psnr_average)



# oooooooooooooooooooooooooooooooooooooooooooooo






