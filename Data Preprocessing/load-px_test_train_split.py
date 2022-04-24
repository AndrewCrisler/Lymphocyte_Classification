# %cd /data/ai_club/team_C
# pip intall spectral --user 
from spectral import *
import numpy as np
from PIL import Image, ImageFile
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True
# Loading spectral image
spectral_img = open_image('ov-63-hd-16ca.hdr')

# Loading masks and converting them to one color channel (True & False)
lymph_mask = np.asarray(Image.open(r"lymphocytes_3_7_22.png").convert("1"))
non_mask = np.asarray(Image.open(r"non_lymphocytes_1_12_22.png").convert("1"))
mask_shape=lymph_mask.shape
#17912, 20292

#Copy Masks (These are for the left and right sections)
lymph_mask_left=lymph_mask.copy()
lymph_mask_right=lymph_mask.copy()
non_mask_left=non_mask.copy()
non_mask_right=non_mask.copy()

#Set masks = 0 for each respective half

slice_fraction=float(5/10)
print(slice_fraction, "Slice Fraction")
#Bad Masks created earlier
'''
lymph_mask_left[(int(mask_shape[0]*slice_fraction)):]=0  #Set the right part of the mask to 0 
lymph_mask_right[:(int(mask_shape[0]*slice_fraction))]=0 # Set the left part of the mask to 0 
non_mask_left[(int(mask_shape[0]*slice_fraction)):]=0  #Set the right part of the mask to 0 
non_mask_right[:(int(mask_shape[0]*slice_fraction))]=0 # Set the left part of the mask to 0 
'''
#good Masks
lymph_mask_left[:,(int(mask_shape[0]*slice_fraction)):]=0  #Set the right part of the mask to 0 
lymph_mask_right[:,:(int(mask_shape[0]*slice_fraction))]=0 # Set the left part of the mask to 0 
non_mask_left[:,(int(mask_shape[0]*slice_fraction)):]=0  #Set the right part of the mask to 0 
non_mask_right[:,:(int(mask_shape[0]*slice_fraction))]=0 # Set the left part of the mask to 0 
# Finding (y,x) tuples for where the mask is True
y_l, x_l = np.where(lymph_mask_left == True)
ny_l, nx_l = np.where(non_mask_left == True)
y_r, x_r = np.where(lymph_mask_right == True)
ny_r, nx_r = np.where(non_mask_right == True)
print(y_l.shape,"y_l")
print( ny_l.shape,"ny_l" )
print(y_r.shape,"y_r")
print( ny_r.shape,"ny_r" )

# Declaring pixel arrays
spectral_result_l = np.empty((len(y_l)+len(ny_l), 395))
spectral_result_r = np.empty((len(y_r)+len(ny_r), 395))
print(spectral_result_l.shape,"Spectral_shape_l")
print(spectral_result_r.shape,"Spectral_shape_r")
# Append the pixels in the spectral image to the pixel array for every (y,x) and (ny,nx) tuple
print("Creating left lymphocyte section of spectral array...")
for index_l, point in enumerate(zip(y_l, x_l)):
    spectral_result_l[index_l] = np.concatenate((np.array([1.0]), spectral_img[point[0], point[1]]), axis=None)
    if(index_l%200 ==0 ): print((index_l/float(len(y_l)))*100, "%") # Progress
print("Creating Right lymphocyte section of spectral array...")
for index_r, point in enumerate(zip(y_r, x_r)):
    spectral_result_r[index_r] = np.concatenate((np.array([1.0]), spectral_img[point[0], point[1]]), axis=None)
    if(index_r%200 ==0 ): print((index_r/float(len(y_r)))*100, "%") # Progress
start_index_l = index_l
start_index_r = index_r
print("Creating Left non-lymphocyte section of spectral array...")
for index_l, point in enumerate(zip(ny_l, nx_l)):
    spectral_result_l[start_index_l + index_l] = np.concatenate((np.array([0.0]), spectral_img[point[0], point[1]]), axis=None)
    if(index_l%200 ==0 ): print((index_l/float(len(ny_l)))*100, "%") # Progress
print("Creating Right non-lymphocyte section of spectral array...")
for index_r, point in enumerate(zip(ny_r, nx_r)):
    spectral_result_r[start_index_r + index_r] = np.concatenate((np.array([0.0]), spectral_img[point[0], point[1]]), axis=None)
