import os
import time

import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from scipy.stats import t


def readData(X_path, Y_Path, min_var=0.5):
    dataX = pd.read_table(X_path, index_col=0, header=0)
    dataY = pd.read_table(Y_Path, index_col=0, header=0)
    dataX = dataX.astype(float)
    dataY = dataY.astype(float)

    dataX = dataX.loc[:,~dataX.columns.duplicated()]
    dataY = dataY.loc[:,~dataY.columns.duplicated()]

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

    # find common samples
    ind = dataY.columns.intersection(dataX.columns)
    #ind = list(set(set(dataX.columns) & set(dataY.columns)))
    # filter to common samples
    dataX = dataX.loc[:, ind]
    dataY = dataY.loc[:, ind]

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
    print('Dataset X dimension  after cleaning: ', dataX.shape)
    print('Dataset Y dimension  after cleaning: ', dataY.shape)
    valuesX = dataX.values
    valuesY = dataY.values
    #dataAll = pd.concat([dataX, dataY])
    featuresX = list(dataX.index)
    featuresY = list(dataY.index)
    #dataAll = dataAll.values

    return valuesX, valuesY, featuresX, featuresY

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


# def btest_corr(dataAll, features, features_y=None, method='spearman', fdr=0.1, Type='X_Y'):
#     corrleationMethod = corrMethod[method]
#     iRow = list(range(0, len(features)))
#     if Type=='X_Y':
#         iCol = list(range(len(features), len(features)+len(features_y)-1))
#         features_y = features + features_y
#     else:
#         features_y = features
#         iCol = list(range(len(features)))
#     tests = []
#     for i in iRow:
#         for j in iCol:
#             if i<=j:
#                 #print(i, j)
#                 X = dataAll[i]
#                 Y = dataAll[j]
#                 nas = np.isnan(X + Y)
#                 not_na = sum(~nas)
#                 #X = Y[~nas]
#                 #Y = Y[~nas]
#                 #new_X, new_Y = remove_pairs_with_a_missing(X, Y)
#                 correlation, pval = corrleationMethod(X, Y)
#                 tests.append([features[i], features_y[j], pval, correlation, not_na])
#     results = pd.DataFrame(tests, columns=['Feature_1','Feature_2','pval', 'Correlation', 'Not_NAs'])
#
#     p_adust, p_threshold = bh(results["pval"].values, fdr)
#
#     results["P_adusted"] = p_adust
#     results["bh_fdr_threshold"] = p_threshold
#     results['Type'] = Type
#     results = results.sort_values(['pval', 'Correlation'],
#                           ascending=[True, False])
#     return results
#
#
# def btest_corr_2(dataAll, features, features_y=None, method='spearman', fdr=0.1, Type='X_Y'):
#     # corrleationMethod = corrMethod[method]
#     iRow = list(range(0, len(features)))
#     if Type == 'X_Y':
#         iCol = list(range(len(features), len(features)+len(features_y)-1))
#         features_y = features + features_y
#     else:
#         features_y = features
#         iCol = list(range(len(features)))
#     tests = []
#
#     # creating the complete dataset
#     dataAll2 = pd.DataFrame(dataAll.T, columns=features_y)
#     cr = dataAll2.corr(method=method)
#
#     # calculating t-statistics, based on the correlations
#     t_stat = (cr*(dataAll2.shape[0]-2)**.5)/(1-cr**2)**.5
#
#     # calculating p-values based on the t-statistics
#     pv = 2 * (1 - t.cdf(abs(t_stat), df=dataAll2.shape[0]-2))
#     #rho_long_format = cr.stack().reset_index().rename(columns={'level_0':'Feature_1','level_1':'Feature_2', 0:'Correlation'})
#     #pval_long_format = pv.stack().reset_index().rename(columns={'level_0':'Feature_1','level_1':'Feature_2', 0:'pval'})
#     #results = pd.concat([pval_long_format, rho_long_format["Correlation"]], axis=1)
#     cr = cr.to_numpy()
#     #pv = pv.to_numpy()
#     for i in iRow:
#         for j in iCol:
#             if i<=j:
#                 #print(i, j)
#                 #print(dataAll[i])
#                 X = dataAll[i]
#                 Y = dataAll[j]
#                 nas = np.isnan(X + Y)
#                 not_na = sum(~nas)
#                 #X = Y[~nas]
#                 #Y = Y[~nas]
#                 #new_X, new_Y = remove_pairs_with_a_missing(X, Y)
#                 correlation = cr[i, j]
#                 pval =  pv[i, j]
#                 tests.append([features[i], features_y[j], pval, correlation, not_na])
#     results = pd.DataFrame(tests, columns=['Feature_1','Feature_2','pval', 'Correlation', 'Not_NAs'])
#
#     p_adust, p_threshold = bh(results["pval"].values, fdr)
#
#     results["P_adusted"] = p_adust
#     results["bh_fdr_threshold"] = p_threshold
#     results['Type'] = Type
#     results = results.sort_values(['pval', 'Correlation'],
#                                   ascending=[True, False])
#     return results


def melter(dat, val):
    dat.reset_index(inplace=True)
    dat.rename({'index': 'Feature_1'}, axis=1, inplace=True)
    dat = pd.melt(dat, id_vars='Feature_1',
                        value_vars=dat[1:],
                        var_name='Feature_2', value_name=val)
    return dat


def btest_corr(dataAll, features, features_y=None, method='spearman', fdr=0.1, Type='X_Y'):
    corrleationMethod = corrMethod[method]
    if Type == 'X_Y':
        features = [f+'_X' for f in features]
        features_y = [f+'_Y' for f in features_y]
        features_y = features + features_y
    else:
        features_y = features

    # creating the complete dataset
    dataAll2 = pd.DataFrame(dataAll.T, columns=features_y)

    t_cr = time.time()
    cr = dataAll2.corr(method=method)
    # print("correlation time: ", time.time()-t_cr)

    t_mask = time.time()
    mask = np.isfinite(dataAll)
    valid_obs = np.zeros((len(features_y), len(features_y)), dtype=np.int32)
    for i in range(len(features_y)):
        valid = mask[0] & mask
        valid = valid.sum(axis=1)
        valid_obs[i,i:] = valid
        valid_obs[i:,i] = valid
        mask = np.delete(mask, 0, 0)

    #print("obs count time: ", time.time()-t_mask)
    valid_obs = pd.DataFrame(valid_obs, columns=features_y, index=features_y)

    # create long dataframes
    # prepare report dataframe
    t_long = time.time()
    check = np.triu(np.ones((dataAll2.shape[1],dataAll2.shape[1])), k=1).astype(bool)
    df_f = cr.where(check).stack().reset_index()
    df_f.columns = ['Feature_1', 'Feature_2', 'Correlation']

    valid_obs = valid_obs.where(check).stack().reset_index()
    df_f.loc[:, 'complete_obs'] = valid_obs.iloc[:, -1]

    # print("long transform time: ", time.time()-t_long)
    # prepare place holders for other values
    df_f.loc[:,'t_statistic'] = None
    df_f.loc[:,'pval'] = None
    df_f.loc[:,'P_adjusted'] = None
    df_f.loc[:,'bh_fdr_threshold'] = None

    # calculate t-statistic based on the correlation and degrees of freedom
    t_pval = time.time()
    df_f.loc[:, 't_statistic'] = (df_f.loc[:, 'Correlation']*
                                 (df_f.loc[:, 'complete_obs']-2)**.5) /\
                                (1-df_f.loc[:, 'Correlation']**2)**.5
    # calculate p-values based on the t-statistic and degrees of freedom
    df_f.loc[:, 'pval'] = 2 * (1 - t.cdf(abs(df_f.loc[:, 't_statistic']),
                                         df=df_f.loc[:, 'complete_obs']-2))
    #print("p-value time: ", time.time()-t_pval)

    # calculate adjusted p-values
    t_bh = time.time()
    p_adust, p_threshold = bh(df_f.loc[:, 'pval'].values, fdr)
    df_f.loc[:, 'P_adjusted'] = p_adust
    df_f.loc[:, 'bh_fdr_threshold'] = p_threshold
    #print("bh time: ", time.time()-t_bh)

    t_names = time.time()
    if Type == 'X_Y':
        df_f.loc[:, 'Type'] = df_f.Feature_1.str[-1]+df_f.Feature_2.str[-2:]
        df_f.loc[:, 'Feature_1'] = df_f.Feature_1.str[:-2]
        df_f.loc[:, 'Feature_2'] = df_f.Feature_2.str[:-2]

    else:
        df_f.loc[:, 'Type'] = Type

    #print("names time: ", time.time()-t_names)

    t_sort = time.time()
    df_f = df_f.sort_values(['pval', 'Correlation'],
                            ascending=[True, False])
    #print("sort time: ", time.time()-t_sort)

    df_f = df_f.sort_values(['pval', 'Correlation'],
                                  ascending=[True, False])

    return df_f

def write_results(results, name, outputpath):
    os.makedirs(outputpath, exist_ok=True)
    if results is not None:
        results.to_csv(outputpath + '/' + name + '.tsv', sep="\t")
        # rho_data = pd.pivot(results, index="Feature_1", columns="Feature_2", values='Correlation') #Reshape from long to wide
        #rho_data.to_csv(outputpath + '/'+name+'.tsv', sep="\t")


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
    if (np.unique(x).shape[0] <= 2 or np.unique(y).shape[0] <=2 ):
        return(0,1)
    corr, pval = spearmanr(x, y)
    return(corr, pval)

def pearson(x, y):
    x, y = remove_missing_values(x, y)
    if (np.unique(x).shape[0] <= 2 or np.unique(y).shape[0] <= 2):
        return(0,1)
    corr, pval = pearsonr(x, y)
    return(corr, pval)
def kendall(x,y):
    x, y = remove_missing_values(x, y)
    if (np.unique(x).shape[0] <= 1 or np.unique(y).shape[0] <= 1):
        return (0, 1)
    corr, pval = kendalltau(x, y)
    return (corr, pval)

corrMethod = {"spearman" : spearman, "pearson": pearson, "kendall":kendall}




