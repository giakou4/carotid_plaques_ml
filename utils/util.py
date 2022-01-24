# -*- coding: utf-8 -*-
"""
@author: Nikolaos Giakoumoglou
@date: Mon Jun  7 13:59:10 2021
"""

import glob

def path2excel(path_feats):
    '''
    Parameters
    ----------
    path_feats : str
        Path where .xlsx file is stored.

    Returns
    -------
    all_names : list
        Names of the .xlsxs files in a list.
    '''
    all_names = []
    cut_value = len(path_feats)
    for f in glob.glob(path_feats + "/*.xlsx"):
        all_names.append(f)
    all_names = [x for x in all_names if not '\y.xlsx' in x]
    for i,name in enumerate(all_names):
        all_names[i] = all_names[i][cut_value:]
        all_names[i] = all_names[i][:-5]
    return all_names


excel2names = {
     'all':'All Features',
     'all_histogram':'All Histogram Features',
     'all_moments':'All Moment Features',
     'all_morphological':'All Morphological Features',
     'all_multiscale':'All Multi-Scale Features',
     'all_textural':'All Textural Features',
     'amfm':'AM-FM',
     'correlogram_d':'Correlogram Distance',
     'correlogram_th':'Correlogram Angle',
     'dwt_bior3.3_levels_3':'DWT',
     'early_texture_all':'Early Textural Features',
     'fdta':'FDTA',
     'fos':'FOS',
     'fps':'FPS',
     'glcm_mean':'GLCM (mean)',
     'glcm_range':'GLCM (range)',
     'glds':'GLDS',
     'glrlm':'GLRLM',
     'glszm':'GLSZM',
     'gt':'GT',
     'histogram':'Histogram',
     'hos_th[135, 140]':'HOS',
     'hu_moments':'Hu Moments',
     'lbp_P[8, 16, 24]_R[1, 2, 3]':'LBP',
     'lte':'LTE',
     'morphological_features_cdf_gray':'Morphological Features - Gray CDF',
     'morphological_features_cdf_H':'Morphological Features - High CDF',
     'morphological_features_cdf_L':'Morphological Features - Low CDF',
     'morphological_features_cdf_M':'Morphological Features - Medium CDF',
     'morphological_features_pdf_gray':'Morphological Features - Gray PDF',
     'morphological_features_pdf_H':'Morphological Features - High PDF',
     'morphological_features_pdf_L':'Morphological Features - Low PDF',
     'morphological_features_pdf_M':'Morphological Features - Medium PDF',
     'multiregion_histogram':'Multi-region Histogram',
     'ngtdm':'NGTDM',
     'sfm':'SFM',
     'shape_parameters':'Shape Parameters',
     'swt_bior3.3_levels_3':'SWT',
     'wp_coif1_maxlevel_3':'WP',
     'zernikes_moments':'Zernikes Moments',
     }
