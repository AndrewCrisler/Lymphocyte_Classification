# %cd /data/ai_club/team_C
# pip intall spectral --user 
from spectral import *
from PIL import Image, ImageFile
import numpy as np
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
    row = []
    for x in range(spectral_shape[1]):
        pixel_arr = spectral_img[y, x]
        standard_pixel_arr = StandardScaler().fit_transform(pixel_arr)
        row.append(standard_pixel_arr)
    pca.partial_fit(row)
    grid.append([row])

# Reducing the dimensionality of the hyperspectral image to 3 principal components
results = None
for row in grid:
    transformed_row = pca.transform(row)
    if results == None:
        results = transformed_row
    else:
        results = np.vstack((results, transformed_row))

# Using Pillow to create a new image with RGB mapping to the 3 principal components 
image = Image.new('RGB', (spectral_shape[0], spectral_shape[1]))
for y in range(spectral_shape[0]):
    for x in range(spectral_shape[1]):
        r, g, b = results[y][x]
        image.putpixel( (x, y), (r, g, b, 255))

# Saving the image
image.save("PCA.png", "PNG")