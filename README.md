# Classification of Carotid Plaques based on Features extracted from Ultrasound Images: A Comparative Study

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/giakou4/carotid_plaques_ml/LICENSE)

## Abstract

The extraction of multiple hand-crafted features from ultrasound images of carotid plaques can be used to effectively differentiate symptomatic and asymptomatic plaques. The aim of this study is to review a wide variety of features sets and evaluate on a common dataset using machine learning methods. Thirty-three feature sets including textural, morphological, histogram-based, multi-scale and moment-based features are extracted from the manually segmented plaque images. For the classification task, the K-Nearest Neighbor (KNN), the Support Vector Machine (SVM) and the Random Forest (RF) machine learning classifiers are used on a Leave-One-Out Cross Validation (LOOCV) policy on the extracted features. The results on a dataset of 85 carotid plaques (41 asymptomatic and 44 symptomatic) shows that the classification is feasible and the highest accuracy achieved is 92.94%.

<p align="center">
  <img src="https://user-images.githubusercontent.com/57758089/157412435-7740562f-c730-41a2-9447-691b509d36a5.png" width="600" height="400">
</p>

<p align="center">
    <em>Figure 1: Examples of asymptomatic (A, B) and symptomatic (C, D) plaques as they were segmented from the expert physician</em>
</p>

## Pre-Requisits

The following packages should be installed:

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

If any of these packages are not installed on your computer, you can install them using the supplied requirements.txt file:  
```pip install -r requirements.txt```

## Files

* ```bag_of_features.py```: Bag of features - K-Means clustering for visual plotting
* ```demo.py```: An example of the feature extraction for a specific plaque
* ```export_features.py```: Export of all the features sets for each plaque with specific parameters
* ```machine_learning.py```: Machine learning classification using KNN, SVM and RF with specific parameters
* ```median_pdf_cdf.py```: Plots the median CDF and PDF of the morphological features with specific parameters
* ```mrmr.py```: Calculates the order of the features significance of the extracted features for each feature set using mRMR algorithm
* ```pca.py```: Performs PCA in 2 dimensions, plots and saves the results
* ```univariate_selection.py```: Performs univariate selection saving the mean and std for each class, the distance of the two classes and the p-value
* ```data```: Due to restriction reasons, dataset is private
    *  ```plaques```: Manually segmented plaques from the expert physician
        * ```7LBW1_plaque.tif```
        * ...
    *  ```points```: Points defining the plaque in the ultrasound as the expert physician pointed out
        * ```7LBW1_points```  
        * ...
    *  ```points_lumen```: A subset of the points above that define neighbor the lumen
        * ```7LBW1_points_lumen``` 
        * ...
    *  ```ultrasounds```: The original normalized and standarized ultrasounds
       * ```7LBW1.bmp``` 
       * ...
    *  ```labels.xlsx``` or ```labels.csv```: The labels of each plaque: symptomatic or asymptomatic
* utils
    * ```Plaque.py```: A class that holds all the information regarding a plaque (ultrasound, points, points near lumen, name) with some useful functions
    * ```util.py```: A file that holds useful functions and plotting utilities
    * ```__init__.py ```

## Material

A total of 85 carotid plaque ultrasound images (41 asymptomatic and 44 symptomatic) producing stenosis in the range of 50% to 99% on duplex scanning are analyzed. The data comrpise a private dataset.

## Pipeline

1. Ultrasound image
2. Preprocessing
    * Manual Normalization by linearly adjusting the image so that the median gray level value of the blood is 0, and the median gray level value of the adventitia (artery wall) is about 190
    * Standarization to 20 pixel/mm
    * Manual plaque segmentation
3. Feature Extraction: 33 feature sets including textural, morphological, histogram-based, multi-scale and moment-based features are extracted from the manually segmented plaque images
4. Dimensionality Reduction
    * Feature Projection: Principal Component Analysis (PCA)
    * Feature Selection: Minimum Rebundancy - Maximum Relevance (mRMR)
    * Raw Features
5. Classification
    * K-Nearest Neighbors (KNN)
    * Support Vector Machines (SVM)
    * Random Forests (RF)

<p align="center">
  <img src="https://user-images.githubusercontent.com/57758089/157421526-f875a946-f962-479b-a2e0-cad3ff8a2a3e.png">
</p>


## Results

Table 1 depicts the results of all features sets from the carotid plaque images. This table displays the features sets used for classification and the highest accuracy observed using raw data, PCA exploiting only the components that contributed to 95\% of the variance in the dataset and top 5 features as denoted from the mRMR algorithm.

<div align="center">
  
| **No** | **Features**           | **Raw** | **PCA** | **mRMR** |
|--------|------------------------|---------|---------|----------|
| 1      | FOS                    | 0.9059  | 0.8941  | 0.9059   |
| 2      | GLCM (mean)            | 0.8471  | 0.8588  | 0.8706   |
| 3      | GLCM (range)           | 0.9059  | 0.8824  | 0.8235   |
| 4      | GLDS                   | 0.8235  | 0.7882  | 0.8235   |
| 5      | NGTDM                  | 0.8941  | 0.8824  | 0.8941   |
| 6      | SFM                    | 0.8353  | 0.8118  | 0.8353   |
| 7      | LTE                    | 0.8235  | 0.7647  | 0.8588   |
| 8      | FDTA                   | 0.8235  | 0.8235  | 0.8235   |
| 9      | GLRLM                  | 0.8941  | 0.8588  | 0.8824   |
| 10     | FPS                    | 0.7765  | 0.8000  | 0.7765   |
| 11     | Shape                  | 0.6824  | 0.6706  | 0.6824   |
| 12     | GLSZM                  | 0.8824  | 0.8824  | 0.8941   |
| 13     | HOS                    | 0.6824  | 0.6824  | 0.6824   |
| 14     | LBP                    | 0.9059  | 0.8941  | 0.8941   |
| 15     | PDF Gray               | 0.8235  | 0.8000  | 0.8000   |
| 16     | CDF Gray               | 0.8471  | 0.8471  | 0.8118   |
| 17     | PDF Low                | 0.8353  | 0.7529  | 0.7529   |
| 18     | CDF Low                | 0.8118  | 0.7882  | 0.7412   |
| 19     | PDF Medium             | 0.6235  | 0.5882  | 0.6118   |
| 20     | CDF Medium             | 0.6118  | 0.6588  | 0.6353   |
| 21     | PDF High               | 0.7765  | 0.7059  | 0.6941   |
| 22     | CDF High               | 0.7647  | 0.7412  | 0.7529   |
| 23     | Histogram              | 0.8824  | 0.8824  | 0.9059   |
| 24     | Multi-region Histogram | 0.9059  | **0.9176**  | **0.9294**   |
| 25     | Correlogram Distance   | 0.8706  | 0.8235  | 0.8118   |
| 26     | Correlogram Angle      | 0.8353  | 0.7529  | 0.6706   |
| 27     | DWT                    | 0.7647  | 0.7412  | 0.7647   |
| 28     | SWT                    | 0.9059  | 0.8118  | 0.8118   |
| 29     | WP                     | 0.8353  | 0.7294  | 0.8000   |
| 30     | GT                     | 0.8941  | 0.8941  | 0.7412   |
| 31     | AM-FM                  | 0.8118  | 0.7412  | 0.8235   |
| 32     | Hu                     | 0.8235  | 0.7882  | 0.8118   |
| 33     | Zernikes               | 0.6824  | 0.6941  | 0.6353   |
| 1-14   | All Textural           | 0.8941  | 0.9059  | 0.8941   |
| 15-22  | All Morphological      | 0.8824  | 0.8471  | 0.8824   |
| 23-26  | All Histogram-based    | 0.8941  | 0.8471  | 0.8471   |
| 27-31  | All Multi-scale        | 0.8588  | 0.8000  | 0.8235   |
| 32-33  | All Moments            | 0.8235  | 0.7294  | 0.7647   |
| 1-33   | All                    | 0.9059  | 0.8588  | 0.5294   |
  
</div>

<p align="center">
    <em> Table 1: Accuracy for each features set as reported from the best classifier</em>
</p>
