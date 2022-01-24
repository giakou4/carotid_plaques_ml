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
import pymrmr
from sklearn.preprocessing import StandardScaler
from utils import path2excel

def options():
    
    parser = argparse.ArgumentParser('arguments for mRMR')
    parser = argparse.ArgumentParser('arguments for bag of features')
    parser.add_argument('--path_features', type=str, default='./results/features/', help='path to load pre-calculated features')
    parser.add_argument('--path_save', type=str, default='./results/mrmr/', help='path to save results')
    opt = parser.parse_args()   
    
    if not os.path.isdir(opt.path_save):
        os.makedirs(opt.path_save)
    if not os.path.isdir(opt.path_features):
        raise Exception("No feature path detected.")  
        
    return opt

if __name__ == "__main__":

    opt = options()
    
    for name in tqdm(path2excel(opt.path_features), desc="mRMR..."):
    
        if name[0:3] in ['all',  'cor', ]:
            continue
        
        # Read data X as DataFrame
        X = pd.read_excel(opt.path_features + name + '.xlsx', engine='openpyxl').iloc[:,1:]
        X = pd.DataFrame(data = StandardScaler().fit_transform(X), columns = X.columns)
    
        # mRMR to data
        results = pymrmr.mRMR(X,'MIQ', X.shape[1])
    
        # Save results to .txt files
        path = opt.path_save + 'mRMR_' + name + '.txt'
        textfile = open(path, "w")
        for element in results:
            textfile.write(element + "\n")
        textfile.close()
