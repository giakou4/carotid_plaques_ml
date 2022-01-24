# -*- coding: utf-8 -*-
"""
@author: Nikolaos Giakoumoglou
@date: Tue May 25 12:36:25 2021
"""

import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
from utils import path2excel, excel2names

def options():
    
    parser = argparse.ArgumentParser('arguments for median cdf and pdf')
    parser.add_argument('--path_features', type=str, default='./results/features/', help='path to load pre-calculated features')
    parser.add_argument('--path_save', type=str, default='./results/median_pdf_cdf/', help='path to save results')
    parser.add_argument('--k', type=int, default=10, help='value of k for KMeans')
    opt = parser.parse_args()   
    
    if not os.path.isdir(opt.path_save):
        os.makedirs(opt.path_save)
    if not os.path.isdir(opt.path_features):
        raise Exception("No feature path detected.")   
        
    return opt


if __name__ == "__main__":

    opt = options()
    
    # Read labels y as numpy array
    y = pd.read_excel(opt.path_features + 'y.xlsx', engine='openpyxl').iloc[:,1:]
    y = pd.DataFrame.to_numpy(y)
    
    for name in tqdm(path2excel(opt.path_features), desc="Calculating median pdfs and cdfs..."):
    
        if name[0:3] not in ['mor']:
            continue
        
        # Read data X as numpy array
        X = pd.read_excel(opt.path_features + name + '.xlsx', engine='openpyxl').iloc[:,1:]
        X = pd.DataFrame.to_numpy(X)
        
        # Split to classes
        X1 = X[(y==0).reshape(len(y==0)),:]
        X2 = X[(y==1).reshape(len(y==1)),:]
        
        # Find median
        X1_med = np.median(X1, axis=0)
        X2_med = np.median(X2, axis=0)
    
        # Plot & save figure as .png file
        plt.plot(X1_med, c='b')
        plt.plot(X2_med, c='r')
        plt.legend(['Asymptomatic', 'Symptomatic'])
        plt.grid()
        tit = excel2names[name]
        plt.title('Median of ' + str(excel2names[name]))
        path = opt.path_save + '_' + name + '.png'
        plt.savefig(path)
        plt.show()
