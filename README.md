# Classification of Carotid Plaques based on Features extracted from Ultrasound Images: A Comparative Study

## Abstract

The extraction of multiple hand-crafted features from ultrasound images of carotid plaques can be used to effectively differentiate symptomatic and asymptomatic plaques. The aim of this study is to review a wide variety of features sets and evaluate on a common dataset using machine learning methods. Thirty-three feature sets including textural, morphological, histogram-based, multi-scale and moment-based features are extracted from the manually segmented plaque images. For the classification task, the K-Nearest Neighbor (KNN), the Support Vector Machine (SVM) and the Random Forest (RF) machine learning classifiers are used on a Leave-One-Out Cross Validation (LOOCV) policy on the extracted features. The results on a dataset of 85 carotid plaques (41 asymptomatic and 44 symptomatic) shows that the classification is feasible and the highest accuracy achieved is 92.94%.

<p align="center">
  <img src="https://user-images.githubusercontent.com/57758089/157412435-7740562f-c730-41a2-9447-691b509d36a5.png" width="300" height="300">
</p>

## Dependencies

* ```numpy==1.19.2```
* ```pandas==1.1.5```
* ```matplotlib==3.3.4```
* ```scikit-image==0.17.2```
* ```scipy==1.5.2```
* ```cv2==3.3.1```
* ```pywt==1.1.1```
* ```mahotas==1.4.11```
* ```researchpy```
* ```sklearn==0.24.1```
* ```pymrmr==0.1.1```
* ```openpyxl==3.0.5```
* ```tqdm==4.56.0```
* ```xlrd==1.2.0```
* ```pyfeats==0.0.11```

## Data

A total of 85 carotid plaque ultrasound images (41 asymptomatic and 44 symptomatic) producing stenosis in the range of 50% to 99% on duplex scanning are analyzed. The data comrpise a private dataset.

## Pipeline

1. Ultrasound image
2. Preprocessing
    * Manual Normalization by linearly adjusting the image so that the median gray level value of the blood is 0, and the median gray level value of the adventitia (artery wall) is about 190
    * Standarization to 20 pixel/mm
    * Manual plaque segmentation
3. Feature Extraction: 33 feature sets including textural, morphological, histogram-based, multi-scale and moment-based features are extracted from the manually segmented plaque images
4. Dimensionality Reduction
    * Principal Component Analysis (PCA)
    * Minimum Rebundancy - Maximum Relevance (mRMR)
    * Raw Features
5. Classification
    * K-Nearest Neighbors (KNN)
    * Support Vector Machines (SVM)
    * Random Forests (RF)

## Results


