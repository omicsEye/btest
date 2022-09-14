import os
import sys
import csv
from scipy.stats import pearsonr, spearmanr
import numpy as np
import pandas as pd
import time

def read_data(X_path, Y_Path):
    dataX = pd.read_table(X_path, index_col=0, header=0)
    dataY = pd.read_table(Y_Path, index_col=0, header=0)

    # find common samples
    ind = dataY.columns.intersection(dataX.columns)

    # filter to common samples
    dataX = dataX.loc[:, ind]
    dataY = dataY.loc[:, ind]

    return dataX, dataY


def btest_corr(df, method='spearman'):
    results = pd.DataFrame()
    Feature_1 = []
    Feature_2 = []
    corrs = []
    p_values = []
    not_nas = []

    if False:
        for i in df.columns:
            for j in df.columns:
                if i != j:
                    try:
                        not_na = sum(df[[i, j]].count(axis=1)==2)
                        #corr = spearmanr(df[i].to_numpy(), df[j].to_numpy(), nan_policy="omit")
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
        rho = df.corr(method='spearman')
        pval = df.corr(method=lambda x, y: spearmanr(x, y, nan_policy='omit')[1]) - np.eye(*rho.shape)
        pval_long_format = pval.stack().reset_index().rename(columns={'level_0':'Feature_1','level_1':'Feature_2', 0:'pval'})
        rho_long_format = rho.stack().reset_index().rename(columns={'level_0':'Feature_1','level_1':'Feature_2', 0:'Correlation'})
        results = pd.concat([pval_long_format, rho_long_format["Correlation"]], axis=1)
        results['Not_NAs'] = 'nan'
    print("Run time: ", rho_excution_time)

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

def corr_paired_data(dataX, dataY, method= 'pearson', fdr=0.1):
    df = pd.concat([dataX, dataY])
    df = df.T
    #rho = df.corr(method='spearman')
    #pval = df.corr(method=lambda x, y: spearmanr(x, y, nan_policy= 'omit')[1]) - np.eye(*rho.shape)
    #rho_long_format = rho.stack().reset_index().rename(columns={'level_0': 'Feature_1','level_1': 'Feature_2', 0: 'Correlation'})
    #pval_long_format = pval.stack().reset_index().rename(columns={'level_0': 'Feature_1','level_1': 'Feature_2', 0: 'P_value'})
    #result = pd.concat([pval_long_format, rho_long_format["Correlation"]], axis=1)


    start_time = time.time()
    results = btest_corr(df)
    rho_excution_time = time.time() - start_time

    within_X = results[results["Feature_1"].isin(dataX.index)]
    within_X = within_X[within_X["Feature_2"].isin(dataX.index)]
    within_X = within_X[within_X["Feature_1"] != within_X["Feature_2"]]

    within_Y = results[results["Feature_1"].isin(dataY.index)]
    within_Y = within_Y[within_Y["Feature_2"].isin(dataY.index)]
    within_Y = within_Y[within_Y["Feature_1"] != within_Y["Feature_2"]]

    X_Y = results[results["Feature_1"].isin(dataX.index) ]
    X_Y = X_Y[X_Y["Feature_2"].isin(dataY.index)]
    #rho_X_Y =  rho[rho.index.isin(dataX.index)]
    #rho_X_Y =  rho_X_Y.loc[:, rho_X_Y.columns.isin(dataY.index)]

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
                          ascending = [True, False])
    within_X = within_X.sort_values(['pval', 'Correlation'],
                                    ascending = [True, False])
    within_Y = within_Y.sort_values(['pval', 'Correlation'],
                                    ascending = [True, False])

    return within_X, within_Y, X_Y, rho_X, rho_Y, rho_X_Y


def write_results(within_X, within_Y, X_Y, rho_X, rho_Y, rho_X_Y, outputpath):
    os.makedirs(outputpath, exist_ok=True)
    X_Y.to_csv(outputpath + '/X_Y.tsv', sep="\t")
    rho_X.to_csv(outputpath + '/simtable_X.tsv', sep="\t")
    rho_X.to_csv(outputpath + '/simtable_Y.tsv', sep="\t")
    rho_X_Y.to_csv(outputpath + '/simtable.tsv', sep="\t")
    within_X.to_csv(outputpath + '/within_X.tsv', sep="\t")
    within_Y.to_csv(outputpath + '/within_Y.tsv', sep="\t")



