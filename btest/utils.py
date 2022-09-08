import os
import sys
import csv
from scipy.stats import pearsonr
import numpy as np
import pandas as pd

def read_data(X_path, Y_Path):
    dataX = pd.read_table(X_path, index_col=0, header=0)
    dataY = pd.read_table(Y_Path, index_col=0, header=0)

    # find common samples
    ind = dataY.index.intersection(dataX.index)

    # filter to common samples
    dataX = dataY.loc[ind, :]
    dataY = dataY.loc[ind, :]

    return dataX, dataY

def corr_paired_data(dataX, dataY, method= 'spearman'):
    results = dataX.corrwith(dataY, axis=1, drop=False, method=method)
    df = pd.concat([dataX, dataY])
    df = df.T
    rho = df.corr()
    pval = df.corr(method=lambda x, y: pearsonr(x, y)[1]) - np.eye(*rho.shape)
    rho_long_format = rho.stack().reset_index().rename(columns={'level_0':'Feature_1','level_1':'Feature_2', 0:'Correlation'})
    pval_long_format = pval.stack().reset_index().rename(columns={'level_0':'Feature_1','level_1':'Feature_2', 0:'P_value'})
    result = pd.concat([pval_long_format, rho_long_format["Correlation"]], axis=1)
    within_X = result[result["Feature_1"].isin(dataX.index) ]
    within_X = within_X[within_X["Feature_2"].isin(dataX.index) ]
    within_Y = result[result["Feature_1"].isin(dataY.index)]
    within_Y = within_Y[within_Y["Feature_2"].isin(dataY.index)]
    X_Y = result[result["Feature_1"].isin(dataX.index) ]
    X_Y = X_Y[X_Y["Feature_2"].isin(dataY.index)
    X_Y_p_adust, X_Y_p_threshold = bh(X_Y["P_value"], .1)
    within_X_p_adust, within_X_p_threshold = bh(within_X["P_value"], .1)
    within_Y_p_adust, within_Y_p_threshold = bh(within_Y["P_value"], .1)
    X_Y["P_adusted"] = X_Y_p_adust
    within_X["P_adusted"] = within_X_p_adust
    within_Y["P_adusted"] = within_Y_p_adust
    X_Y["bh_fdr_threshold"] = X_Y_p_threshold
    within_X["bh_fdr_threshold"] = within_X_p_threshold
    within_Y["bh_fdr_threshold"] = within_Y_p_threshold
    X_Y = X_Y.sort_values(['P_value', 'Correlation'],
                          ascending = [True, False])
    within_X = within_X.sort_values(['P_value', 'Correlation'],
                                    ascending = [True, False])
    within_Y = within_Y.sort_values(['P_value', 'Correlation'],
                                    ascending = [True, False])
    return within_X, within_Y, X_Y


def write_results(outputpath):
    os.makedirs(outputpath, exist_ok=True)
    X_Y.to_csv(outputpath + '/X_Y.tsv', sep="\t")
    rho_X = rho.iloc[0:500, 0:500]
    rho_X.to_csv(outputpath + '/simtable_X.tsv', sep="\t")
    rho.to_csv(outputpath + '/simtable.tsv', sep="\t")
    within_X.to_csv(outputpath + '/within_X.tsv', sep="\t")
    within_Y.to_csv(outputpath + '/within_Y.tsv', sep="\t")

def fdr(p_vals):
    from scipy.stats import rankdata
    ranked_p_values = rankdata(p_vals, method='ordinal')
    fdr = p_vals * len(p_vals) / ranked_p_values
    fdr[fdr > 1] = 1
    return fdr


def bh(p, q):
    from scipy.stats import rankdata
    pRank = rankdata(p, method='ordinal')
    m = len(p)
    p_adust = p *  len(p) / pRank
    max_i = 0
    for i in range(len(p)):
        if p_adust[i] <= q and max_i <= pRank[i]:
            max_i = pRank[i]
            p_threshold = p[i]
    return p_adust, p_threshold

