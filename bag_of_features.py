# -*- coding: utf-8 -*-
"""
@author: Nikolaos Giakoumoglou
@date: Mon Jun  7 23:15:18 2021
"""

import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from utils import path2excel, excel2names

def options():
    
    parser = argparse.ArgumentParser('arguments for bag of features')
    parser.add_argument('--path_features', type=str, default='./results/features/', help='path to load pre-calculated features')
    parser.add_argument('--path_save', type=str, default='./results/bag_of_features/', help='path to save results')
    parser.add_argument('--k', type=int, default=10, help='value of k for KMeans')
    opt = parser.parse_args()   
    
    if not os.path.isdir(opt.path_save):
        os.makedirs(opt.path_save)
    if not os.path.isdir(opt.path_features):
        raise Exception("No feature path detected.")  
        
    return opt

if __name__ == "__main__":
    
    opt = options() 
    
    # Read labels as DataFrame
    y = pd.read_excel(opt.path_features + 'y.xlsx', engine='openpyxl').iloc[:,1:]

    for name in tqdm(path2excel(opt.path_features), desc="bag of features..."):
        
        # Read data X as DataFrame and scale them using Standard Scaler
        X = pd.read_excel(opt.path_features + name + '.xlsx', engine='openpyxl').iloc[:,1:]
        X = StandardScaler().fit_transform(X)
    
        # Find 1st class & perform KMean
        X1 = X[y['Label']==1,:]
        kmeans = KMeans(n_clusters=opt.k)
        kmeans.fit_predict(X1)
        labels1 = kmeans.labels_
    
        # Find 2nd class & perform KMean
        X2 = X[y['Label']==0,:]
        kmeans = KMeans(n_clusters=opt.k)
        kmeans.fit_predict(X2)
        labels2 = kmeans.labels_
    
        # Plot & save results in .png files
        fig, (ax1, ax2) = plt.subplots(2)
        ax1.hist(labels1, color='r')
        ax1.set_ylabel('Symptomatic')
        ax2.hist(labels2, color='b')
        ax2.set_ylabel('Asymptomatic')
        plt.suptitle('Bag of Features for ' + excel2names[name])
        plt.savefig(opt.path_save+'histogram_' + str(name) + '.png')
        plt.show()