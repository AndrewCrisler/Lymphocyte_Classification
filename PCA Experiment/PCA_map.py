%cd /data/ai_club/team_C
from spectral import *
from PIL import Image, ImageFile
import numpy as np
import cv2
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import IncrementalPCA
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Loading spectral image
spectral_img = open_image('ov-63-hd-16ca.hdr')

# Instantiating incremental PCA
pca = IncrementalPCA(n_components=3)

# Standardizing and chunking the spectral image by row and fitting sklearn's partial PCA
spectral_shape = spectral_img.shape
grid = []
for y in range(spectral_shape[0]):
    if y % 500 == 0:
        print(f"PCA fit: {y}")
    row = spectral_img[y, :, :][0]
    pca.partial_fit(row)
    grid.append(row)
    
# Reducing the dimensionality of the hyperspectral image to 3 principal components
results = None
for i, row in enumerate(grid):
    if i % 200 == 0:
        print(f"PCA transform: {i}")
    transformed_row = pca.transform(row)
    if results is None:
        results = transformed_row
    else:
        results = np.vstack((results, transformed_row))

# Reshaping results array
results_t = np.reshape(results, (spectral_shape[0], spectral_shape[1], 3))

# Normalizing and scaling results between 0 and 255
PC_2d_Norm = np.zeros((spectral_shape[0], spectral_shape[1], 3))
for i in range(3):
    PC_2d_Norm[:,:,i] = cv2.normalize(results_t[:,:,i], np.zeros((spectral_shape[0], spectral_shape[1], 3)), 0, 255, cv2.NORM_MINMAX)

# Using Pillow to create a new image with RGB mapping to the 3 principal components 
image = Image.new('RGB', (spectral_shape[1], spectral_shape[0]))
for y in range(spectral_shape[0]):
    for x in range(spectral_shape[1]):
        r, g, b = PC_2d_Norm[y][x]
        image.putpixel((x, y), (np.uint8(r), np.uint8(g), np.uint8(b), 255))

# Saving the image
image.save("Full_PCA.png", "PNG")