import os
import sys
import csv
from scipy.stats import pearsonr, spearmanr, kendalltau
import numpy as np
import pandas as pd
import time
from numpy import array
import shutil
import time
import math
import random
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from btest import config
import itertools


def readData(X_path, Y_Path):
    dataX = pd.read_table(X_path, index_col=0, header=0)
    dataY = pd.read_table(Y_Path, index_col=0, header=0)
    dataX = dataX.astype(float)
    dataY = dataY.astype(float)
    # find common samples
    ind = dataY.columns.intersection(dataX.columns)
    #ind = list(set(set(dataX.columns) & set(dataY.columns)))
    # filter to common samples
    dataX = dataX.loc[:, ind]
    dataY = dataY.loc[:, ind]

    dataX = dataX.loc[:,~dataX.columns.duplicated()]
    dataY = dataY.loc[:,~dataY.columns.duplicated()]

    return dataX, dataY

def dataProcess(dataX, dataY, min_var =0.0):

    l1_before = len(dataX.columns)
    l2_before = len(dataY.columns)

    # remove samples/columns with all NaN/missing values
    ## First change missing value to np.NaN for pandas
    dataX[dataX == ''] = np.NAN
    dataY[dataY == ''] = np.NAN
    dataX = dataX.dropna(axis=1, how='all')
    dataY = dataY.dropna(axis=1, how='all')

    l1_after = len(dataX.columns)
    l2_after = len(dataY.columns)

    # replace np.NaN's with 'NaN'
    #dataX[dataX.isnull()] = 'NaN'
    #dataY[dataY.isnull()] = 'NaN'

    if l1_before > l1_after:
        print("--- %d samples/columns with all missing values have been removed from the first dataset " % (
                l1_before - l1_after))

    if l2_before > l2_after:
        print("--- %d samples/columns with all missing values have been removed from the second dataset " % (
                l2_before - l2_after))
    l1_before = len(dataX.index)
    l2_before = len(dataY.index)
    dataX = dataX[dataX.var(axis=1) > min_var]
    dataY = dataY[dataY.var(axis=1) > min_var]

    l1_after = len(dataX.index)
    l2_after = len(dataY.index)
    if l1_before > l1_after:
        print("--- %d features with variation equal or less than %.3f have been removed from the first dataset " % (
            l1_before - l1_after, min_var))

    if l2_before > l2_after:
        print(
            "--- %d features with variation equal or less than %.3f have been removed from the second dataset " % (
                l2_before - l2_after, min_var))
    #valuesX = dataX.values
    #valuesY = dataY.values
    dataAll = pd.concat([dataX, dataY])
    featuresX = list(dataX.index)
    featuresY = list(dataY.index)
    dataAll = dataAll.values

    return dataAll, featuresX, featuresY

def btest_corr_pandas(df, method='spearman', pval=False):
    results = pd.DataFrame()
    Feature_1 = []
    Feature_2 = []
    corrs = []
    p_values = []
    not_nas = []

    if True:
        for i in df.columns:
            for j in df.columns:
                if i != j:
                    try:
                        not_na = sum(df[[i, j]].count(axis=1)==2)
                        corr = spearmanr(df[i].to_numpy(), df[j].to_numpy(), nan_policy="omit")
                        #corr = stats.kendalltau(ac, bc)
                    except ValueError:
                        corr = ('nan', 'nan')
                        not_na = 'nan'
                    Feature_1.append(i)
                    Feature_2.append(j)
                    corrs.append(corr[0])
                    p_values.append(corr[1])
                    not_nas.append(not_na)
        results['Feature_1'] = Feature_1
        results['Feature_2'] = Feature_2
        results['pval'] = p_values
        results['Correlation'] = corrs
        results['Not_NAs'] = not_nas
    else:
        # for i in df.columns:
        #     for j in df.columns:
        #         try:
        #             not_na = sum(df[[i, j]].count(axis=1)==2)
        #         except ValueError:
        #             not_na = 'nan'
        #         not_nas.append(not_na)
        if pval:
            rho = df.corr(method='spearman')
            pval = df.corr(method=lambda x, y: spearmanr(x, y, nan_policy='omit')[1]) - np.eye(*rho.shape)
            pval_long_format = pval.stack().reset_index().rename(columns={'level_0':'Feature_1','level_1':'Feature_2', 0:'pval'})
            rho_long_format = rho.stack().reset_index().rename(columns={'level_0':'Feature_1','level_1':'Feature_2', 0:'Correlation'})
            results = pd.concat([pval_long_format, rho_long_format["Correlation"]], axis=1)
            results['Not_NAs'] = 'nan'
        else:
            rho = df.corr(method='spearman')
            rho_long_format = rho.stack().reset_index().rename(columns={'level_0':'Feature_1','level_1':'Feature_2', 0:'Correlation'})
            results = rho_long_format
            results['pval'] = 'nan'
            results['Not_NAs'] = 'nan'

    return results


def fdr(p_vals):
    from scipy.stats import rankdata
    ranked_p_values = rankdata(p_vals, method='ordinal')
    fdr = p_vals * len(p_vals) * 1.0 / (ranked_p_values * 1.0)
    fdr[fdr > 1.0] = 1.0
    return fdr


def bh(p, q):
    from scipy.stats import rankdata
    pRank = rankdata(p, method='ordinal')
    m = len(p)
    p_adust = p * len(p) * 1.0 / (pRank * 1.0)
    max_i = 0
    p_threshold = p[0]
    for i in range(len(p)):
        if p_adust[i] <= q and max_i <= pRank[i]:
            max_i = pRank[i]
            p_threshold = p[i]
    return p_adust, p_threshold


def btest_corr(dataAll, featuresX, featuresY, method='spearman', fdr=0.1):
    corrleationMethod = corrMethod[method]
    iRow = len(dataAll)
    iCol = len(dataAll)
    tests = []
    features = featuresX + featuresY
    for i in list(range(iRow)):
        for j in list(range(iCol)):
            if i<=j:
                X = dataAll[i]
                Y=  dataAll[j]
                nas = np.logical_or(X != X, Y != Y)
                not_na = sum(~nas)
                #X = Y[~nas]
                #Y = Y[~nas]
                #new_X, new_Y = remove_pairs_with_a_missing(X, Y)
                correlation, pval = corrleationMethod(X, Y)
                tests.append([features[i],features[j],pval, correlation, not_na])
    results = pd.DataFrame(tests, columns = ['Feature_1','Feature_2','pval', 'Correlation', 'Not_NAs'])
    return results

def corr_paired_data(dataAll, featuresX, featuresY, method='spearman', fdr=0.1):
    start_time = time.time()
    results = btest_corr(dataAll, featuresX, featuresY, method=method)

    within_X = results[results["Feature_1"].isin(featuresX)]
    within_X = within_X[within_X["Feature_2"].isin(featuresX)]
    within_X = within_X[within_X["Feature_1"] != within_X["Feature_2"]]

    within_Y = results[results["Feature_1"].isin(featuresY)]
    within_Y = within_Y[within_Y["Feature_2"].isin(featuresY)]
    within_Y = within_Y[within_Y["Feature_1"] != within_Y["Feature_2"]]

    X_Y = results[~np.logical_or(results["Feature_1"].isin(featuresY), results["Feature_2"].isin(featuresX))]

    rho_X_Y = pd.pivot(X_Y, index="Feature_1", columns = "Feature_2",values = 'Correlation') #Reshape from long to wide
    rho_X = pd.pivot(within_X, index="Feature_1", columns = "Feature_2",values = 'Correlation') #Reshape from long to wide
    rho_Y = pd.pivot(within_Y, index="Feature_1", columns = "Feature_2",values = 'Correlation') #Reshape from long to wide

    within_X_p_adust, within_X_p_threshold = bh(within_X["pval"].values, fdr)
    within_Y_p_adust, within_Y_p_threshold = bh(within_Y["pval"].values, fdr)
    X_Y_p_adust, X_Y_p_threshold = bh(X_Y["pval"].values, fdr)

    within_X["P_adusted"] = within_X_p_adust
    within_X["bh_fdr_threshold"] = within_X_p_threshold

    within_Y["P_adusted"] = within_Y_p_adust
    within_Y["bh_fdr_threshold"] = within_Y_p_threshold

    X_Y["P_adusted"] = X_Y_p_adust
    X_Y["bh_fdr_threshold"] = X_Y_p_threshold

    X_Y = X_Y.sort_values(['pval', 'Correlation'],
                          ascending=[True, False])
    within_X = within_X.sort_values(['pval', 'Correlation'],
                                    ascending=[True, False])
    within_Y = within_Y.sort_values(['pval', 'Correlation'],
                                    ascending=[True, False])

    rho_excution_time = time.time() - start_time
    print("Run time: ", rho_excution_time)
    return within_X, within_Y, X_Y, rho_X, rho_Y, rho_X_Y


def write_results(within_X, within_Y, X_Y, rho_X, rho_Y, rho_X_Y, outputpath):
    os.makedirs(outputpath, exist_ok=True)
    X_Y.to_csv(outputpath + '/X_Y.tsv', sep="\t")
    rho_X.to_csv(outputpath + '/simtable_X.tsv', sep="\t")
    rho_Y.to_csv(outputpath + '/simtable_Y.tsv', sep="\t")
    rho_X_Y.to_csv(outputpath + '/simtable.tsv', sep="\t")
    within_X.to_csv(outputpath + '/within_X.tsv', sep="\t")
    within_Y.to_csv(outputpath + '/within_Y.tsv', sep="\t")


def remove_missing_values(x, y):
    '''Given x and y all in numpy arrays, remove pairs that contain missing values
    '''
    # nan != nan = TRUE
    nas = np.logical_or(x != x, y != y)
    return(x[~nas], y[~nas])
def pearson(x, y):
    x, y = remove_missing_values(x, y)
    if (np.unique(x).shape[0] == 1 or np.unique(y).shape[0] == 1):
        return(0,1)
    corr, pval = pearsonr(x, y)
    return(corr, pval)

def spearman(x, y):
    x, y = remove_missing_values(x, y)
    if (np.unique(x).shape[0] == 1 or np.unique(y).shape[0] == 1):
        return(0,1)
    corr, pval = spearmanr(x, y)
    return(corr, pval)

def pearson(x, y):
    x, y = remove_missing_values(x, y)
    if (np.unique(x).shape[0] == 1 or np.unique(y).shape[0] == 1):
        return(0,1)
    corr, pval = pearsonr(x, y)
    return(corr, pval)
def kendall(x,y):
    x, y = remove_missing_values(x, y)
    if (np.unique(x).shape[0] == 1 or np.unique(y).shape[0] == 1):
        return (0, 1)
    corr, pval = kendalltau(x, y)
    return (corr, pval)

corrMethod = {"spearman" : spearman, "pearson": pearson, "kendall":kendall}




