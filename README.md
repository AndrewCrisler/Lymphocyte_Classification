# Lymphocyte_Classification

Abstract:
Current methods for diagnosing the progression of multiple types of cancer within patients rely on interpreting stained needle biopsies. This process is time-consuming and susceptible to error throughout the paraffinization, Hematoxylin and Eosin (H&E) staining, deparaffinization, and annotation stages. Fourier Transform Infrared (FTIR) imaging has been shown to be a promising alternative to staining for appropriately annotating biopsy cores without the need for deparaffinization or H&E staining with the use of Fourier Transform Infrared (FTIR) images when combined with machine learning to interpret the dense spectral information. We present a machine learning pipeline to segment white blood cell (lymphocyte) pixels in hyperspectral images of biopsy cores. These cells are clinically important for diagnosis, but some prior work has struggled to incorporate them due to difficulty obtaining precise pixel labels. Evaluated methods include Support Vector Machine (SVM), Gaussian Naïve Bayes, and Multilayer Perceptron (MLP), as well as analyzing the comparatively modern convolutional neural network (CNN).

Classical model instructions:


CNN instructions:
Run load_tile_px_3_8_test_train_split.py if changes to annotations are needed. 

Run/edit Good CNN Good Data (Split).ipynb

