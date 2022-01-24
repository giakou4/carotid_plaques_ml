import numpy as np
import pandas as pd
from tqdm import tqdm
import cv2
import pyfeats

from utils import Plaque

#%% Path & Name of Plaque     
path = './data/'
labels = pd.read_excel(path+'labels.xlsx')
path_features = './results/features/'
IMG_NO = len(labels)

#%% Parameters
perc = 1                    # Percentage of the plaque to take into consideration when calculating features in (0,1]
Dx             = [0,1,1,1]  # A. Early Textural - GLDS
Dy             = [1,1,0,-1] # A. Early Textural  - GLDS
d              = 1          # A. Early Textural  - NGTDM
Lr, Lc         = 4, 4       # A. Early Textural  - SFM
l              = 7          # A. Early Textural  - LTE
s              = 4          # A. Early Textural  - FDTA
th             = [135,140]  # A. Late Textural - HOS
P              = [8,16,24]  # A. Late Textural - LBP
R              = [1,2,3]    # A. Late Textural - LBP
N              = 30         # B Morphology
bins_hist      = 32         # C. Histogram - All
num_eros       = 3          # C. Histogram - Multi-region Histogram
square_size    = 3          # C. Histogram - Multi-region Histogram
wavelet_dwt    = 'bior3.3'  # D. Multi-Scale - DWT
wavelet_swt    = 'bior3.3'  # D. Multi-Scale - SWT
wavelet_wp     = 'coif1'    # D. Multi-Scale -  WP
levels_dwt     = 3          # D. Multi-Scale - DWT
levels_swt     = 3          # D. Multi-Scale - SWT
levels_wp      = 3          # D. Multi-Scale - WP
bins_digitize  = 32         # C. Histogram - Correlogram
bins_hist_corr = 32         # C. Histogram - Correlogram
zernikes_radii = 9          # E. Other - Zernikes Moments

#%% Init arrays
names = []

# A. Textural
np_fos = np.zeros((IMG_NO,16), np.double)
np_glcm_mean = np.zeros((IMG_NO,14), np.double)
np_glcm_range = np.zeros((IMG_NO,14), np.double)
np_glds = np.zeros((IMG_NO,5), np.double)
np_ngtdm = np.zeros((IMG_NO,5), np.double)
np_sfm = np.zeros((IMG_NO,4), np.double)
np_lte = np.zeros((IMG_NO,6), np.double)
np_fdta = np.zeros((IMG_NO,s+1), np.double)
np_glrlm = np.zeros((IMG_NO,11), np.double)
np_fps = np.zeros((IMG_NO,2), np.double)
np_shape_parameters = np.zeros((IMG_NO,5), np.double)
np_hos = np.zeros((IMG_NO,len(th)), np.double)
np_lbp = np.zeros((IMG_NO,len(P)*2), np.double)
np_glszm = np.zeros((IMG_NO,14), np.double)

# B. Morphological
pdf_L = np.zeros((IMG_NO,N), np.double)
pdf_M = np.zeros((IMG_NO,N), np.double)
pdf_H = np.zeros((IMG_NO,N), np.double)
cdf_L = np.zeros((IMG_NO,N), np.double)
cdf_M = np.zeros((IMG_NO,N), np.double)
cdf_H = np.zeros((IMG_NO,N), np.double)
pdf_gray = np.zeros((IMG_NO,N), np.double)
cdf_gray = np.zeros((IMG_NO,N), np.double)

# C. Histogram
np_histogram = np.zeros((IMG_NO,bins_hist), np.double)
np_multiregion_histogram = np.zeros((IMG_NO,bins_hist*num_eros), np.double)
np_correlogram_d = np.zeros((IMG_NO,bins_digitize*bins_hist), np.double)
np_correlogram_th = np.zeros((IMG_NO,bins_digitize*bins_hist), np.double)

# D. Multi-Scale
np_dwt = np.zeros((IMG_NO,6*levels_dwt), np.double)
np_swt = np.zeros((IMG_NO,6*levels_swt), np.double)
np_wp = np.zeros((IMG_NO,(4**levels_wp-1)*2), np.double)
np_gt = np.zeros((IMG_NO,16), np.double)
np_amfm = np.zeros((IMG_NO,32*4), np.double)

# E. Other
np_hu =  np.zeros((IMG_NO,7), np.double)
np_zernikes =  np.zeros((IMG_NO,25), np.double)

#%% Calculate Features
progress = tqdm(range(0,IMG_NO), desc="Calculating Textural Features...")
for i in progress:
    name = labels.iloc[i,0]
    names.append(name)
    
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

    plaque = Plaque(ultrasound, points, points_lumen, name, pad=2)
    plaque.mask = Plaque.get_perc_ROI(plaque.mask, plaque.perimeter_lumen, perc)
       
    # A. Textural 
    progress.set_description('Calculating Early Textural Features' + ' for ' + name)
    np_fos[i,:], labels_fos = pyfeats.fos(plaque.plaque, plaque.mask)
    np_glcm_mean[i,:], np_glcm_range[i,:], labels_glcm_mean, labels_glcm_range = pyfeats.glcm_features(plaque.plaque, ignore_zeros=True)
    np_glds[i,:], labels_glds = pyfeats.glds_features(plaque.plaque, plaque.mask, Dx=Dx, Dy=Dy)
    np_ngtdm[i,:], labels_ngtdm = pyfeats.ngtdm_features(plaque.plaque, plaque.mask, d=d)
    np_sfm[i,:], labels_sfm = pyfeats.sfm_features(plaque.plaque, plaque.mask, Lr=Lr, Lc=Lc)
    np_lte[i,:], labels_lte = pyfeats.lte_measures(plaque.plaque, plaque.mask, l=l)
    np_fdta[i,:], labels_fdta = pyfeats.fdta(plaque.plaque, plaque.mask, s=s) 
    np_glrlm[i,:], labels_glrlm = pyfeats.glrlm_features(plaque.plaque, plaque.mask, Ng=256)
    np_fps[i,:], labels_fps = pyfeats.fps(plaque.plaque, plaque.mask)
    np_shape_parameters[i,:], labels_shape_parameters = pyfeats.shape_parameters(plaque.plaque, plaque.mask, plaque.perimeter, pixels_per_mm2=1)
    progress.set_description('Calculating Late Textural Features')
    np_hos[i,:], labels_hos = pyfeats.hos_features(plaque.plaque, th=th)
    np_lbp[i,:], labels_lbp = pyfeats.lbp_features(plaque.plaque, plaque.mask, P=P, R=R)
    np_glszm[i,:], labels_glszm = pyfeats.glszm_features(plaque.plaque, plaque.mask)
    
    # B. Morphological
    progress.set_description('Calculating Morphological Features' + ' for ' + name)
    pdf_gray[i,:], cdf_gray[i,:] = pyfeats.grayscale_morphology_features(plaque.plaque, N=N)
    pdf_L[i,:], pdf_M[i,:], pdf_H[i,:], cdf_L[i,:], cdf_M[i,:], cdf_H[i,:] = \
            pyfeats.multilevel_binary_morphology_features(plaque.plaque, plaque.mask, N=N)

    # C. Histogram
    progress.set_description('Calculating Histogram Features' + ' for ' + name)
    np_histogram[i,:], labels_histogram = pyfeats.histogram(plaque.plaque, plaque.mask, bins_hist)
    np_multiregion_histogram[i,:], labels_multiregion_histogram = pyfeats.multiregion_histogram(plaque.plaque, plaque.mask, bins=bins_hist, num_eros=num_eros,square_size=square_size)
    np_correlogram_d[i,:], np_correlogram_th[i,:], labels_correlogram = pyfeats.correlogram(plaque.plaque, plaque.mask, bins_digitize=bins_digitize, bins_hist=bins_hist, flatten=True)

    # D. Multi-Scale
    progress.set_description('Calculating Multi-Scale Features' + ' for ' + name)
    np_dwt[i,:], labels_dwt = pyfeats.dwt_features(plaque.plaque, plaque.mask, wavelet=wavelet_dwt, levels=levels_dwt)
    np_swt[i,:], labels_swt = pyfeats.swt_features(plaque.plaque, plaque.mask, wavelet=wavelet_swt, levels=levels_swt)
    np_wp[i,:], labels_wp = pyfeats.wp_features(plaque.plaque, plaque.mask, wavelet=wavelet_wp, maxlevel=levels_wp)
    np_gt[i,:], labels_gt = pyfeats.gt_features(plaque.plaque, plaque.mask)
    np_amfm[i,:], labels_amfm = pyfeats.amfm_features(plaque.plaque)
    
    # E. Other
    progress.set_description('Calculating Other Features' + ' for ' + name)
    np_hu[i,:], labels_hu = pyfeats.hu_moments(plaque.plaque)
    np_zernikes[i,:], labels_zernikes = pyfeats.zernikes_moments(plaque.plaque, zernikes_radii)
    
    
#%% Convert to pandas

# A. Early Textural
df_fos = pd.DataFrame(data=np_fos, index=names, columns=labels_fos)
df_glcm_mean = pd.DataFrame(data=np_glcm_mean, index=names, columns=labels_glcm_mean)
df_glcm_range = pd.DataFrame(data=np_glcm_range, index=names, columns=labels_glcm_range)
df_glds = pd.DataFrame(data=np_glds, index=names, columns=labels_glds)
df_ngtdm = pd.DataFrame(data=np_ngtdm, index=names, columns=labels_ngtdm)
df_sfm = pd.DataFrame(data=np_sfm, index=names, columns=labels_sfm)
df_lte = pd.DataFrame(data=np_lte, index=names, columns=labels_lte)
df_fdta = pd.DataFrame(data=np_fdta, index=names, columns=labels_fdta)
df_glrlm = pd.DataFrame(data=np_glrlm, index=names, columns=labels_glrlm)
df_fps = pd.DataFrame(data=np_fps, index=names, columns=labels_fps)
df_shape_parameters = pd.DataFrame(data=np_shape_parameters, index=names, columns=labels_shape_parameters)
df_early_texture_all = pd.concat([df_fos, df_glcm_mean, df_glcm_range, df_glds,
                                 df_ngtdm, df_sfm, df_lte, df_fdta, df_glrlm, df_fps,
                                 df_shape_parameters], axis=1)
df_hos = pd.DataFrame(data=np_hos, index=names, columns=labels_hos)
df_lbp = pd.DataFrame(data=np_lbp, index=names, columns=labels_lbp)
df_glszm = pd.DataFrame(data=np_glszm, index=names, columns=labels_glszm)

# B Morphological
pdf_cdf_labels = ['N_'+str(i) for i in range(N)]
df_pdf_gray = pd.DataFrame(data=pdf_gray, index=names, columns=['pdf_gray_'+ s for s in pdf_cdf_labels])
df_pdf_L = pd.DataFrame(data=pdf_L, index=names, columns=['pdf_L_'+ s for s in pdf_cdf_labels])
df_pdf_M = pd.DataFrame(data=pdf_M, index=names, columns=['pdf_M_'+ s for s in pdf_cdf_labels])
df_pdf_H = pd.DataFrame(data=pdf_H, index=names, columns=['pdf_H_'+ s for s in pdf_cdf_labels])
df_cdf_gray = pd.DataFrame(data=cdf_gray, index=names, columns=['cdf_gray_'+ s for s in pdf_cdf_labels])
df_cdf_L = pd.DataFrame(data=cdf_L, index=names, columns=['cdf_L_'+ s for s in pdf_cdf_labels])
df_cdf_M = pd.DataFrame(data=cdf_M, index=names, columns=['cdf_M_'+ s for s in pdf_cdf_labels])
df_cdf_H = pd.DataFrame(data=cdf_H, index=names, columns=['cdf_H_'+ s for s in pdf_cdf_labels])

# C. Histogram
df_histogram = pd.DataFrame(data=np_histogram, index=names, columns=labels_histogram)
df_multiregion_histogram = pd.DataFrame(data=np_multiregion_histogram, index=names, columns=labels_multiregion_histogram)
df_correlogram_d =pd.DataFrame(data=np_correlogram_d, index=names, columns=[s+'_distance' for s in labels_correlogram])
df_correlogram_th = pd.DataFrame(data=np_correlogram_th, index=names, columns=[s+'_theta' for s in labels_correlogram])

# D. Multi-Scale
df_dwt = pd.DataFrame(data=np_dwt, columns=labels_dwt, index=names)
df_swt = pd.DataFrame(data=np_swt, columns=labels_swt, index=names)
df_wp = pd.DataFrame(data=np_wp, columns=labels_wp, index=names)
df_gt = pd.DataFrame(data=np_gt, columns=labels_gt, index=names)
df_amfm = pd.DataFrame(data=np_amfm, columns=labels_amfm, index=names)

# E. Other
df_hu = pd.DataFrame(data=np_hu, columns=labels_hu, index=names)
df_zernikes = pd.DataFrame(data=np_zernikes, columns=labels_zernikes, index=names)

#%% Save with pandas
try:   
    # A. Early Textural
    df_fos.to_excel(path_features+'fos.xlsx', index = True)
    df_glcm_mean.to_excel(path_features+'glcm_mean.xlsx', index = True)
    df_glcm_range.to_excel(path_features+'glcm_range.xlsx', index = True)
    df_glds.to_excel(path_features+'glds.xlsx', index = True)
    df_ngtdm.to_excel(path_features+'ngtdm.xlsx', index = True)
    df_sfm.to_excel(path_features+'sfm.xlsx', index = True)
    df_lte.to_excel(path_features+'lte.xlsx', index = True)
    df_fdta.to_excel(path_features+'fdta.xlsx', index = True)
    df_glrlm.to_excel(path_features+'glrlm.xlsx', index = True)
    df_fps.to_excel(path_features+'fps.xlsx', index = True)
    df_shape_parameters.to_excel(path_features+'shape_parameters.xlsx', index = True)
    df_early_texture_all.to_excel(path_features+'early_texture_all.xlsx', index = True)
    df_hos.to_excel(path_features+'hos_th'+str(th)+'.xlsx', index = True)
    df_lbp.to_excel(path_features+'lbp_P'+str(P)+'_R'+str(R)+'.xlsx', index = True)
    df_glszm.to_excel(path_features+'glszm.xlsx', index = True)
    
    # B. Morphological
    df_pdf_gray.to_excel(path_features+'morphological_features_pdf_gray.xlsx', index = True)
    df_pdf_L.to_excel(path_features+'morphological_features_pdf_L.xlsx', index = True)
    df_pdf_M.to_excel(path_features+'morphological_features_pdf_M.xlsx', index = True)
    df_pdf_H.to_excel(path_features+'morphological_features_pdf_H.xlsx', index = True)
    df_cdf_gray.to_excel(path_features+'morphological_features_cdf_gray.xlsx', index = True)
    df_cdf_L.to_excel(path_features+'morphological_features_cdf_L.xlsx', index = True)
    df_cdf_M.to_excel(path_features+'morphological_features_cdf_M.xlsx', index = True)
    df_cdf_H.to_excel(path_features+'morphological_features_cdf_H.xlsx', index = True)
    
    # C. Histogram
    df_histogram.to_excel(path_features+'histogram.xlsx', index = True)
    df_multiregion_histogram.to_excel(path_features+'multiregion_histogram.xlsx', index = True)
    df_correlogram_d.to_excel(path_features+'correlogram_d.xlsx', index = True)
    df_correlogram_th.to_excel(path_features+'correlogram_th.xlsx', index = True)
    
    # D. Multi-Scale
    df_dwt.to_excel(path_features+'dwt_'+str(wavelet_dwt)+'_levels_'+str(levels_dwt)+'.xlsx', index = True)
    df_swt.to_excel(path_features+'swt_'+str(wavelet_swt)+'_levels_'+str(levels_swt)+'.xlsx', index = True)
    df_wp.to_excel(path_features+'wp_'+str(wavelet_wp)+'_maxlevel_'+str(levels_wp)+'.xlsx', index = True) 
    df_gt.to_excel(path_features+'gt.xlsx', index = True)
    df_amfm.to_excel(path_features+'amfm.xlsx', index = True)
    
    # E. Other
    df_hu.to_excel(path_features+'hu_moments.xlsx', index = True)
    df_zernikes.to_excel(path_features+'zernikes_moments.xlsx', index = True)
        
    print('\nData was successfully saved')
except:
    print("\nAn exception occured")

