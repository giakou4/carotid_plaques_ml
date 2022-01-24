# -*- coding: utf-8 -*-
"""
@author: Nikolaos Giakoumoglou
@date: Sun May 23 22:45:54 2021
"""

import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from utils import path2excel, excel2names

def options():
    
    parser = argparse.ArgumentParser('arguments for PCA')
    parser.add_argument('--path_features', type=str, default='./results/features/', help='path to load pre-calculated features')
    parser.add_argument('--path_save', type=str, default='./results/pca/', help='path to save results')
    opt = parser.parse_args()   
    
    if not os.path.isdir(opt.path_save):
        os.makedirs(opt.path_save)
    if not os.path.isdir(opt.path_features):
        raise Exception("No feature path detected.")
        
    return opt


if __name__ == "__main__":

    opt = options()
    
    # Read labels y as DataFrame
    y = pd.read_excel(opt.path_features + 'y.xlsx', engine='openpyxl').iloc[:,1:]
    
    for name in tqdm(path2excel(opt.path_features), desc="PCA 2d..."):
        
        if name[0:3] in ['all',  'cor', ]:
            continue
        
        # Read data X as DataFrame & scale them using Standard Scaler
        X = pd.read_excel(opt.path_features + name + '.xlsx', engine='openpyxl').iloc[:,1:]
        X = StandardScaler().fit_transform(X)
    
        # Apply PCA to data in 2 dimensions
        pca = PCA(n_components=2)
        pcs = pca.fit_transform(X)
        components = pd.DataFrame(data = pcs, columns = ['PC1', 'PC2'])
        components = pd.concat([components, y], axis = 1)
    
        # Plot components in 2 dimensions & save figure as .png file
        fig = plt.figure(figsize = (8,8))
        ax = fig.add_subplot(1,1,1)
        ax.set_xlabel('Principal Component 1', fontsize = 15)
        ax.set_ylabel('Principal Component 2', fontsize = 15)
        ax.set_title('2 component PCA with ' + str(pca.explained_variance_ratio_) + 'explained variance ratio', fontsize = 20)
        targets = [0, 1]
        targets_full_name = ['Asymptomatic', 'Symptomatic']
        colors = ['r', 'b']
        for target, color in zip(targets,colors):
            indicesToKeep = components['Label'] == target
            ax.scatter(components.loc[indicesToKeep, 'PC1'], components.loc[indicesToKeep, 'PC2'], c = color, s = 50)
        ax.legend(targets_full_name)
        ax.grid()
        plt.title('PCA in 2 dimensions of ' + str(excel2names[name]))
        path = opt.path_save+'pca2d_'+name+'.png'
        plt.savefig(path)
        plt.show()
