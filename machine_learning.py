# -*- coding: utf-8 -*-
"""
Author: Nikolaos Giakoumoglou
Date: Mon Dec  6 11:07:38 2021
"""
import os
import pandas as pd
import numpy as np
import argparse
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import LeaveOneOut, GridSearchCV
from sklearn.decomposition import PCA
import pymrmr
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics.pairwise import chi2_kernel
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import cross_val_predict
from utils import path2excel, excel2names
import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")


def options():
    
    parser = argparse.ArgumentParser('arguments for machine learning')
    parser.add_argument('--path_features', type=str, default='./results/features/', help='path to load pre-calculated features')
    parser.add_argument('--path_save', type=str, default='./results/median_pdf_cdf/', help='path to save results')
    parser.add_argument('--no_mrmr', type=int, default=5, help='mRMR top features')
    parser.add_argument('--explained_var_ratio', type=int, default=0.95, help='ratio of variance for PCA')
    parser.add_argument('--n_jobs', type=int, default=-1, help='number of jobs for cross-validation experiments')
    opt = parser.parse_args()   
    
    if not os.path.isdir(opt.path_save):
        os.makedirs(opt.path_save)
    if not os.path.isdir(opt.path_features):
        raise Exception("No feature path detected.")   
        
    return opt

    
def get_PCA(X_np, opt):
    X_ss = StandardScaler().fit_transform(X_np)
    pca = PCA(opt.explained_var_ratio)
    X_pca = pca.fit_transform(X_ss)
    return X_pca

def get_mRMR(X_df, opt):
    X_np = pd.DataFrame.to_numpy(X_df)
    no_samples, no_features = X_np.shape
    if no_features < opt.no_mrmr:
        opt.no_mrmr = no_features
    res = pymrmr.mRMR(pd.DataFrame(StandardScaler().fit_transform(X_df),columns = X_df.columns),'MIQ', opt.no_mrmr)
    X_mrmr = np.zeros((no_samples, opt.no_mrmr), np.float64)
    for j, r in enumerate(res):
        X_mrmr[:,j] = X_df[r]
    X_mrmr = MinMaxScaler().fit_transform(X_mrmr)
    opt.no_mrmr = 5 # restart value
    return X_mrmr

if __name__ == "__main__":
    
    opt = options()
    
    names = path2excel(opt.path_features)
        
    for name_id, name in enumerate(names):
        
        full_name = excel2names[name]
               
        print('\nFeature set No. {}/{} - {}'.format(name_id+1, len(names), full_name))
        
        X_df = pd.read_excel(opt.path_features + name + '.xlsx', engine='openpyxl').iloc[:,1:]
        y_df = pd.read_excel(opt.path_features + 'y.xlsx', engine='openpyxl').iloc[:,1:]
        
        X_np = pd.DataFrame.to_numpy(X_df)
        X_np = np.nan_to_num(X_np)
        y = pd.DataFrame.to_numpy(y_df).ravel()
                
        X_minmax = MinMaxScaler().fit_transform(X_np)
        X_ss = StandardScaler().fit_transform(X_np)
        X_pca = get_PCA(X_np, opt)
        X_mrmr = get_mRMR(X_df, opt)
        
        Xs = [X_minmax, X_pca, X_mrmr]
        Xs_names = ['Raw Data (MinMax)', 'PCA {}% variance'.format(opt.explained_var_ratio*100), 'mRMR top {} features'.format(opt.no_mrmr)]
        
        param_grid_svm = [{'kernel': ['rbf'], 'gamma': [10, 1, 0.1, 0.01, 0.001], 'C': [0.1, 1, 10, 100]}, {'kernel': ['linear'], 'C': [0.1, 1, 10, 100]}]
        param_grid_knn = [{'n_neighbors': list(range(1,21)), 'weights': ['uniform','distance']}]
        param_grid_rf = dict(max_depth=[2, 8, 16], n_estimators=[64, 128, 256], max_features=['sqrt','log2'], min_samples_split = [2], min_samples_leaf = [1])
        params = [param_grid_svm, param_grid_knn, param_grid_rf]
        
        cv = LeaveOneOut()
        scoring = {"Accuracy": make_scorer(accuracy_score)}
        refit_scorer = "Accuracy"
        
        clfs = [SVC(), KNeighborsClassifier(), RandomForestClassifier()]
        clfs_names = ["SVM", "KNN", "RF"]
          
        for ii, X in enumerate(Xs):      
            for clf, param in zip(clfs, params):  
                if name in ['histogram', 'multiregion_histogram', 'correlogram_d', 'correlogram_th'] and clf==SVC():
                    param = [{'kernel': ['precomputed']}]
                    K = chi2_kernel(X,X,gamma=1)
                    grid_obj = GridSearchCV(clf, param, cv=cv, scoring=scoring, refit=refit_scorer, return_train_score=True, verbose=0, n_jobs=opt.n_jobs)
                    grid_fit = grid_obj.fit(K, y)
                else:
                    grid_obj = GridSearchCV(clf, param, cv=cv, scoring=scoring, refit=refit_scorer, return_train_score=True, verbose=0, n_jobs=opt.n_jobs)
                    grid_fit = grid_obj.fit(X, y)
                best_clf = grid_fit.best_estimator_
                best_params = grid_fit.best_params_
                results = grid_fit.cv_results_
                best_index = np.nonzero(results["mean_test_%s" % refit_scorer] == grid_fit.best_score_)[0][0]
                grid_acc = results["mean_test_Accuracy"][best_index]
                #print('Accuracy={:.4f} with {} & {}'.format(grid_acc, clfs_names[clfs.index(clf)], Xs_names[ii]))
                y_pred = cross_val_predict(best_clf, X, y, cv=cv)
                acc = accuracy_score(y, y_pred)
                prec = precision_score(y, y_pred)
                rec = recall_score(y, y_pred)
                f1 = f1_score(y, y_pred)
                if type(clf) == type(SVC()):
                    best_clf_2 = SVC(**best_params, probability=True)
                elif type(clf) == type(KNeighborsClassifier()):
                    best_clf_2 = KNeighborsClassifier(**best_params)
                elif type(clf) == type(RandomForestClassifier()):
                    best_clf_2 = RandomForestClassifier(**best_params)                       
                y_score = cross_val_predict(best_clf_2, X, y, cv=cv, method='predict_proba')
                #y_pred = np.argmax(y_score, axis=-1)
                AUC = roc_auc_score(y, y_score[:, 1])
                #print('Accuracy={:.4f} | Precision={:.4f} | Recall={:.4f} | F1-score={:.4f} | AUC={:.4f}\n'.format(acc, prec, rec, f1, AUC))
                print('Accuracy={:.4f} with {} & {} | Accuracy={:.4f} | Precision={:.4f} | Recall={:.4f} | F1-score={:.4f} | AUC={:.4f}'.format(grid_acc, clfs_names[clfs.index(clf)], Xs_names[ii], acc, prec, rec, f1, AUC))        