# -*- coding: utf-8 -*-
"""
@author: Nikolaos Giakoumoglou
@date: Sun May 16 11:18:40 2021
"""

#%% Imports
import pandas as pd
import numpy as np
import cv2
from utils import Plaque
import pyfeats
        
#%% Path & Name of Plaque  
path = './data/'
labels = pd.read_excel(path+'labels.xlsx', engine='openpyxl')
idx = 41
name = labels.iloc[idx,0]

# Load ultrasound
path_ultrasound = path + 'ultrasounds\\' + name + '.bmp'
ultrasound = cv2.imread(path_ultrasound, cv2.IMREAD_GRAYSCALE)

# Load points
path_points = path + 'points\\' + name + '_points.out'
points = np.loadtxt(path_points, delimiter=',')
points = np.array(points, np.int32)

# Load points near lumen
path_points_lumen = path + 'points_lumen\\' + name + '_points_lumen.out'
points_lumen = np.loadtxt(path_points_lumen, delimiter=',')
points_lumen = np.array(points_lumen, np.int32)

#%% Load Plaque, show some basic things
plaque = Plaque(ultrasound, points, points_lumen, name, pad=2)
plaque.plotUnderlinedPlaque()
plaque.plotPlaque()
plaque.plotPerimeter()
plaque.plotMask()
plaque.plotHistogram(bins=32)
plaque.plotGeroulakosClassification()

#%% A1. Texture features
features = {}
features['A_FOS'] = pyfeats.fos(plaque.plaque, plaque.mask)
features['A_GLCM'] = pyfeats.glcm_features(plaque.plaque, ignore_zeros=True)
features['A_GLDS'] = pyfeats.glds_features(plaque.plaque, plaque.mask, Dx=[0,1,1,1], Dy=[1,1,0,-1])
features['A_NGTDM'] = pyfeats.ngtdm_features(plaque.plaque, plaque.mask, d=1)
features['A_SFM'] = pyfeats.sfm_features(plaque.plaque, plaque.mask, Lr=4, Lc=4)
features['A_LTE'] = pyfeats.lte_measures(plaque.plaque, plaque.mask, l=7)
features['A_FDTA'] = pyfeats.fdta(plaque.plaque, plaque.mask, s=3)
features['A_GLRLM'] = pyfeats.glrlm_features(plaque.plaque, plaque.mask, Ng=256)
features['A_FPS'] = pyfeats.fps(plaque.plaque, plaque.mask)
features['A_Shape_Parameters'] = pyfeats.shape_parameters(plaque.plaque, plaque.mask, plaque.perimeter, pixels_per_mm2=1)
features['A_HOS'] = pyfeats.hos_features(plaque.plaque, th=[135,140])
features['A_LBP'] = pyfeats.lbp_features(plaque.plaque, plaque.mask, P=[8,16,24], R=[1,2,3])
features['A_GLSZM'] = pyfeats.glszm_features(plaque.plaque, plaque.mask)
pyfeats.plot_sinogram(plaque.plaque, plaque.name)

#%% B. Morphological features
features['B_Morphological_Grayscale_pdf'], features['B_Morphological_Grayscale_cdf'] = pyfeats.grayscale_morphology_features(plaque.plaque, N=30)
features['B_Morphological_Binary_L_pdf'], features['B_Morphological_Binary_M_pdf'], features['B_Morphological_Binary_H_pdf'], features['B_Morphological_Binary_L_cdf'], features['B_Morphological_Binary_M_cdf'], features['B_Morphological_Binary_H_cdf'] = pyfeats.multilevel_binary_morphology_features(plaque.plaque, plaque.mask, N=30)

pyfeats.plot_pdf_cdf(features['B_Morphological_Grayscale_pdf'], features['B_Morphological_Grayscale_cdf'], plaque.name)
pyfeats.plot_pdfs_cdfs(features['B_Morphological_Binary_L_pdf'], features['B_Morphological_Binary_M_pdf'], features['B_Morphological_Binary_H_pdf'], features['B_Morphological_Binary_L_cdf'], features['B_Morphological_Binary_M_cdf'], features['B_Morphological_Binary_H_cdf'])

#%% C. Histogram Based features
features['C_Histogram'] = pyfeats.histogram(plaque.plaque, plaque.mask, 32)
features['C_MultiregionHistogram'] = pyfeats.multiregion_histogram(plaque.plaque, plaque.mask, bins=32, num_eros=3, square_size=3)
features['C_Correlogram'] = pyfeats.correlogram(plaque.plaque, plaque.mask, bins_digitize=32, bins_hist=32, flatten=True)

pyfeats.plot_histogram(plaque.plaque, plaque.mask, bins=32, name=plaque.name)
pyfeats.plot_correlogram(plaque.plaque, plaque.mask, bins_digitize=32, bins_hist=32, name=plaque.name)

#%% D. Multi-Scale features
features['D_DWT'] = pyfeats.dwt_features(plaque.plaque, plaque.mask, wavelet='bior3.3', levels=3)
features['D_SWT'] = pyfeats.swt_features(plaque.plaque, plaque.mask, wavelet='bior3.3', levels=3)
features['D_WP'] = pyfeats.wp_features(plaque.plaque, plaque.mask, wavelet='coif1', maxlevel=3)
features['D_GT'] = pyfeats.gt_features(plaque.plaque, plaque.mask)
features['D_AMFM'] = pyfeats.amfm_features(plaque.plaque)

#%% E. Other
#features['E_HOG'] = hog_features(plaque.plaque)
features['E_HuMoments'] = pyfeats.hu_moments(plaque.plaque)
#features['E_TAS'] = tas_features(plaque.plaque)
features['E_ZernikesMoments'] = pyfeats.zernikes_moments(plaque.plaque, radius=9)

#%% Print
for x, y in features.items():
    if x.startswith("B") | x.startswith("C"):
        continue
    print('-' * 50)
    print(x)
    print('-' * 50)
    if len(y[1]) == 1:
        print(y[1][0], '=', y[0])
    else:
        for i in range(len(y[1])):
           print(y[1][i], '=', y[0][i])