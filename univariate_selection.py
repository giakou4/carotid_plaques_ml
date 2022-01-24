# -*- coding: utf-8 -*-
"""
@author: Nikolaos Giakoumoglou
@date: Sun May 23 22:45:35 2021
"""

import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse
import researchpy as rp
import math
from utils import path2excel

def options():
    
    parser = argparse.ArgumentParser('arguments for t-test')
    parser.add_argument('--path_features', type=str, default='./results/features/', help='path to load pre-calculated features')
    parser.add_argument('--path_save', type=str, default='./results/ttest/', help='path to save results')
    opt = parser.parse_args()       
    
    if not os.path.isdir(opt.path_save):
        os.makedirs(opt.path_save)
    if not os.path.isdir(opt.path_features):
        raise Exception("No feature path detected.")    
        
    return opt


if __name__ == "__main__":

    opt = options()
    
    # Read labels y as DataFrame    
    y = pd.read_excel(opt.path_features+'y.xlsx').iloc[:,1:]
    
    for name in tqdm.tqdm(path2excel(opt.path_features), desc="univariate selection..."):
        
        # Read data X as DataFrame and combine them with y in df as DataFrame
        X = pd.read_excel(opt.path_features+name+'.xlsx').iloc[:,1:]
        df = pd.concat([X, y], axis=1)
        
        results = []
        for col in X.columns:
            
            # Split data to classes
            group1 = df[col][df['Label'] == 1]
            group2 = df[col][df['Label'] == 0]
    
            # Perform t-test to each class
            res = rp.ttest(group1=group1, group1_name= 1,
                           group2=group2, group2_name= 0)
            mean1 = res[0].iloc[0,2]
            std1 = res[0].iloc[0,3]
            mean2 = res[0].iloc[1,2]
            std2 = res[0].iloc[1,3]
            pvalue = res[1].iloc[3,:]['results']
            dis = abs(mean1-mean2)/math.sqrt(std1**2+std2**2+1e-16)
            results.append([mean1, std1, mean2, std2, dis, pvalue])
    
        # Save results in .xlsx files
        results = np.array(results)
        df_results = pd.DataFrame(results,columns=['mean1','std1','mean2','std2',
                                                   'distance','p-value'],
                                  index=X.columns)
        df_results.to_excel(opt.path_save + name + '_ttest.xlsx', index = True)
