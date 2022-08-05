######parsin files#######
#!/usr/bin/env python
'''
Parses input/output formats,
manages transformations
'''

import csv
import re
import sys
from numpy import array

import numpy as np
import pandas as pd
from pandas import *
from . import config
from . import store
from . import stats


def wrap_features(txt, width=40):
    '''helper function to wrap text for long labels'''
    import textwrap
    txt = txt.replace('s__', '').replace('g__', '').replace('f__', '').replace('o__', '').replace('c__', '').replace(
        'p__', '').replace('k__', '')
    txt = str(txt).split("|")
    txt = [val for val in txt if len(val) > 0]
    if len(txt) > 1:
        txt = txt[len(txt) - 2] + " " + txt[len(txt) - 1]
    else:
        txt = txt[0]
    return txt  # '\n'.join(textwrap.wrap(txt, width))


def substitute_special_characters(txt):
    txt = re.sub('[\n\;]', '_', txt).replace('__', '_').replace('__', '_').replace('_', ' ')  # replace('.','_')
    return txt


def load(file):
    # Read in the file
    if isinstance(file, pd.DataFrame):
        return file.values
    try:
        import io
        file_handle = io.open(file, encoding='utf-8')
    except EnvironmentError:
        sys.exit("Error: Unable to read file: " + file)

    csvr = csv.reader(file_handle, dialect="excel-tab")  # csv.excel_tab,

    # Ignore comment lines in input file
    data = []
    comments = []
    for line in csvr:
        # Add comment to list
        if re.match("#", line[0]):
            comments.append(line)
        else:
            # First data line found
            data = [line]
            break

    # Check if last comment is header
    if comments:
        header = comments[-1]
        # if the same number of columns then last comment is header
        if len(header) == len(data[0]):
            data = [header, data[0]]

    # finish processing csv
    for line in csvr:
        data.append(line)

    # close csv file
    file_handle.close()

    return np.array(data)


class Input:
    """

    Parser class for input

    Handles missing values, data type transformations

    * `CON` <- continous
    * `CAT` <- categorical
    * `BIN` <- binary
    * `LEX` <- lexical

    """

    def __init__(self, strFileName1, strFileName2=None, var_names=True, headers=False):

        # Data types
        self.continuous = "CON"
        self.categorical = "CAT"
        self.binary = "BIN"
        self.lexical = "LEX"

        # Boolean indicators
        self.varNames = var_names
        self.headers = headers

        # Initialize data structures
        self.strFileName1 = strFileName1
        self.strFileName2 = strFileName1 if strFileName2 is None else strFileName2

        self.discretized_dataset1 = None
        self.discretized_dataset2 = None
        self.orginal_dataset1 = None
        self.orginal_dataset2 = None
        self.outName1 = None
        self.outName2 = None

        self.outType1 = None
        self.outType2 = None

        self.outHead1 = None
        self.outHead2 = None

        self._load()
        self._parse()
        self._filter_to_common_columns()
        print(
            "Discretizing is started using: %s style for filtering features with low entropy!" % config.strDiscretizing)
        self._discretize()
        self._remove_low_entropy_features()
        if len(self.outName1) < 2 or len(self.outName1) < 2:
            sys.exit(
                "--- btest to continue needs at lease two features in each dataset!!!\n--- Please repeat the one feature or provide the -a AllA option in the command line to do pairwise alla-against-all test!!")
        store.smart_decisoin()
        if store.bypass_discretizing():
            # try:
            #print(self.orginal_dataset1)
            #print(self.orginal_dataset2)
            self.orginal_dataset1 = np.asarray(self.orginal_dataset1, dtype=float)

            self.orginal_dataset2 = np.asarray(self.orginal_dataset2, dtype=float)
            self._transform_data()
            # self.discretized_dataset1 = self.orginal_dataset1
            # self.discretized_dataset2 = self.orginal_dataset2
           # except:
                #sys.exit("--- Please check your data types and your similarity metric!")
        self._check_for_semi_colon()

    def get(self):

        return [(self.discretized_dataset1, self.orginal_dataset1, self.outName1, self.outType1, self.outHead1),
                (self.discretized_dataset2, self.orginal_dataset2, self.outName2, self.outType2, self.outHead2)]

    def _load(self):
        self.orginal_dataset1 = load(self.strFileName1)
        self.orginal_dataset2 = load(self.strFileName2)

    def _check_for_semi_colon(self):
        # check the names of features that btest uses to make sure they don't have ; which
        # is special character to separate features in output files
        for i in range(len(self.outName1)):
            if ";" in self.outName1[i]:
                print("Feature names warning!")
                print(self.outName1[i])
                sys.exit("In the first dataset, your feature (row) names contains ; which is the special character btest uses for separating features,\n \
				             Please replace it with another character such as _ or use --format-feature-names option in command line which replaces all special characters by _")
        for i in range(len(self.outName2)):
            if ";" in self.outName2[i]:
                print("Feature names warning!")
                print(self.outName2[i])
                sys.exit("In the second dataset, your feature (row) names contains ; which is the special character btest uses for separating features,\n \
				             Please replace it with another character such as _ or use --format-feature-names option in command line which replaces all special characters by _")

    def _discretize(self):
        self.discretized_dataset1 = stats.discretize(self.orginal_dataset1, style=config.strDiscretizing,
                                                     data_type=config.data_type[0])
        self.discretized_dataset2 = stats.discretize(self.orginal_dataset2, style=config.strDiscretizing,
                                                     data_type=config.data_type[1])

    def _parse(self):
        def __parse(pArray, bVar, bHeaders):

            aOut = []
            aNames = []
            used_names = []
            aTypes = []
            aHeaders = None

            # Parse header if indicated by user or "#"
            if bHeaders or re.match("#", str(pArray[0, 0])):
                aHeaders = list(pArray[0, 1:])
                pArray = pArray[1:]

            # Parse variable names
            if bVar:
                aNames = list(pArray[:, 0])
                aNames = list(map(str, aNames))
                if config.format_feature_names:
                    aNames = list(map(wrap_features, aNames))
                    aNames = list(map(substitute_special_characters, aNames))
                pArray = pArray[:, 1:]

            # replace missing charaters with nan
            # pArray[pArray == config.missing_char] = 'NaN'

            # print pArray
            # # Parse data types, missing values, and whitespace
            if config.missing_method:
                from sklearn.impute import SimpleImputer
                imp = SimpleImputer(missing_values=config.missing_char, strategy=config.missing_method)
            # imp.fit(pArray)

            for i, line in enumerate(pArray):
                # *   If the line is not full,  replace the Nones with nans                                           *
                # *****************************************************************************************************
                # line = list(map(lambda x: 'NaN' if x == config.missing_char else x, line))  ###### np.nan Convert missings to nans
                if all([val == config.missing_char for val in line]):
                    # if all values in a feature are missing values then skip the feature
                    print('All missing value in', aNames[i])
                    continue
                if not aNames:
                    aNames.append(i)
                # aOut.append(line)
                try:
                    if config.missing_method:
                        line = array(imp.fit_transform(line.reshape(1, -1)))[0]
                    aTypes.append("CON")
                except ValueError:
                    line = line  # we are forced to conclude that it is implicitly categorical, with some lexical ordering
                    aTypes.append("LEX")

                used_names.append(aNames[i])
                aOut.append(line)
            # if there is categorical data then do btest with AllA style of
            # finding the BH threshold using all p-values
            if "LEX" in aTypes:
                config.do_pair_block = True

            return aOut, used_names, aTypes, aHeaders

        self.orginal_dataset1, self.outName1, self.outType1, self.outHead1 = __parse(self.orginal_dataset1,
                                                                                     self.varNames, self.headers)
        self.orginal_dataset2, self.outName2, self.outType2, self.outHead2 = __parse(self.orginal_dataset2,
                                                                                     self.varNames, self.headers)
        config.data_type[0] = self.outType1
        config.data_type[1] = self.outType2

    def _filter_to_common_columns(self):
        """
        Make sure that the data are well-formed
        """

        assert (len(self.orginal_dataset1) == len(self.outType1))
        assert (len(self.orginal_dataset2) == len(self.outType2))

        if self.outName1:
            assert (len(self.orginal_dataset1) == len(self.outName1))
        if self.outName2:
            assert (len(self.orginal_dataset2) == len(self.outName2))
        if self.outHead1:
            assert (len(self.orginal_dataset1[0]) == len(self.outHead1))
        if self.outHead2:
            assert (len(self.orginal_dataset2[0]) == len(self.outHead2))

        # If sample names are included in headers in both files,
        # check that the samples are in the same order
        if self.outHead1 and self.outHead2:
            header1 = "\t".join(self.outHead1)
            header2 = "\t".join(self.outHead2)
            # print header1, header2
            # if not (header1.lower() == header2.lower()):
            # +
            # "." + " \n File1 header: " + header1 + "\n" +
            # " File2 header: " + header2)
            try:
                df1 = pd.DataFrame(self.orginal_dataset1, index=self.outName1, columns=self.outHead1)
            except:
                df1 = pd.DataFrame(self.orginal_dataset1, index=self.outName1, columns=self.outHead1)
            try:
                df2 = pd.DataFrame(self.orginal_dataset2, index=self.outName2, columns=self.outHead2)
            except:
                df2 = pd.DataFrame(self.orginal_dataset2, index=self.outName2, columns=self.outHead2)
            # print df1.columns.isin(df2.columns)
            # print df2.columns.isin(df1.columns)

            l1_before = len(df1.columns)
            l2_before = len(df2.columns)

            # remove samples/columns with all NaN/missing values
            # First change missing value to np.NaN for pandas
            df1[df1 == config.missing_char] = np.NAN
            df2[df2 == config.missing_char] = np.NAN
            df1 = df1.dropna(axis=1, how='all')
            df2 = df2.dropna(axis=1, how='all')
            l1_after = len(df1.columns)
            l2_after = len(df2.columns)
            # replace np.NaN's with 'NaN'
            df1[df1.isnull()] = 'NaN'
            df2[df2.isnull()] = 'NaN'

            if l1_before > l1_after:
                print("--- %d samples/columns with all missing values have been removed from the first dataset " % (
                            l1_before - l1_after))

            if l2_before > l2_after:
                print("--- %d samples/columns with all missing values have been removed from the second dataset " % (
                            l2_before - l2_after))

            # Keep common samples/columns between two data frame
            paired_sample = list(set(set(df1.columns) & set(df2.columns)))
            #print(len(paired_sample), paired_sample)
            #print(df1.shape, df2.shape)
            df1 = df1.loc[:, paired_sample]
            df2 = df2.loc[:, paired_sample]
            df1 = df1.loc[:,~df1.columns.duplicated()]
            df2 = df2.loc[:,~df2.columns.duplicated()]

            # reorder df1 columns as the columns order of df2

            #df1[paired_sample]
            #df2[paired_sample]
            #print(df1.shape, df1, df2.shape, df2)
            self.orginal_dataset1 = df1.values
            self.orginal_dataset2 = df2.values

            # print self.orginal_dataset1
            # print HSIC.HSIC_pval(df1.values,df2.values, p_method ='gamma', N_samp =1000)

            self.outName1 = list(df1.index)
            self.outName2 = list(df2.index)
            # print self.outName1
            # print self.outName2
            # self.outType1 = int
            # self.outType2 = int

            # self.outHead1 = df1.columns
            # self.outHead2 = df2.columns
            self.outHead1 = df1.columns
            self.outHead2 = df2.columns
            print(
                ("The program uses %s common samples between the two data sets based on headers") % (str(df1.shape[1])))
        if len(self.orginal_dataset1[0]) != len(self.orginal_dataset2[0]):
            print (len(df1.columns), len(df2.columns))
            sys.exit("Have you provided --header option to use sample/column names for shared sample/columns?")

    def _remove_low_variant_features(self):
        try:
            df1 = pd.DataFrame(self.orginal_dataset1, index=self.outName1, columns=self.outHead1, dtype=float)
        except:
            df1 = pd.DataFrame(self.orginal_dataset1, index=self.outName1, columns=self.outHead1, dtype=float)
        try:
            df2 = pd.DataFrame(self.orginal_dataset2, index=self.outName2, columns=self.outHead2, dtype=float)
        except:
            df2 = pd.DataFrame(self.orginal_dataset2, index=self.outName2, columns=self.outHead2, dtype=float)
        # print df1.columns.isin(df2.columns)
        # print df2.columns.isin(df1.columns)
        # print df1.var(), np.var(df2, axis=1)
        l1_before = len(df1.index)
        l2_before = len(df2.index)
        df1 = df1[df1.var(axis=1) > config.min_var]
        df2 = df2[df2.var(axis=1) > config.min_var]

        l1_after = len(df1.index)
        l2_after = len(df2.index)
        if l1_before > l1_after:
            print("--- %d features with variation equal or less than %.3f have been removed from the first dataset " % (
            l1_before - l1_after, config.min_var))

        if l2_before > l2_after:
            print(
                "--- %d features with variation equal or less than %.3f have been removed from the second dataset " % (
                l2_before - l2_after, config.min_var))
        # reorder df1 columns as the columns order of df2
        # df1 = df1.loc[:, df2.columns]

        self.orginal_dataset1 = df1.values
        self.orginal_dataset2 = df2.values
        # print self.orginal_dataset1
        self.outName1 = list(df1.index)
        self.outName2 = list(df2.index)
        # print self.outName1
        # self.outType1 = int
        # self.outType2 = int

        # self.outHead1 = df1.columns
        # self.outHead2 = df2.columns
        # print self.outHead1
        # print df2
        assert (len(self.orginal_dataset1[0]) == len(self.orginal_dataset2[0]))

    def _remove_low_entropy_features(self):
        # print self.discretized_dataset1
        # print self.orginal_dataset1
        df1 = pd.DataFrame(self.discretized_dataset1, index=self.outName1, columns=self.outHead1)
        df1_org = pd.DataFrame(self.orginal_dataset1, index=self.outName1, columns=self.outHead1)

        df2 = pd.DataFrame(self.discretized_dataset2, index=self.outName2, columns=self.outHead2)
        df2_org = pd.DataFrame(self.orginal_dataset2, index=self.outName2, columns=self.outHead2)

        # print df1.columns.isin(df2.columns)
        # print df2.columns.isin(df1.columns)
        # print df1.var(), np.var(df2, axis=1)
        l1_before = len(df1.index)
        l2_before = len(df2.index)

        # filter for only features with entropy greater or equal that the threshold
        temp_df1 = df1
        df1 = df1[df1.apply(stats.get_enropy, 1) >= config.entropy_threshold1]
        df1_org = df1_org[temp_df1.apply(stats.get_enropy, 1) > config.entropy_threshold1]

        temp_df2 = df2
        df2 = df2[df2.apply(stats.get_enropy, 1) >= config.entropy_threshold2]
        df2_org = df2_org[temp_df2.apply(stats.get_enropy, 1) >= config.entropy_threshold2]

        l1_after = len(df1.index)
        l2_after = len(df2.index)
        if l1_before > l1_after:
            print("--- %d features with entropy equal or less than %.3f have been removed from the first dataset " % (
            (l1_before - l1_after), config.entropy_threshold1))

        if l2_before > l2_after:
            print("--- %d features with entropy equal or less than %.3f have been removed from the second dataset " % (
            (l2_before - l2_after), config.entropy_threshold2))
        # reorder df1 columns as the columns order of df2
        # df1 = df1.loc[:, df2.columns]

        self.discretized_dataset1 = df1.values
        self.orginal_dataset1 = df1_org.values

        self.discretized_dataset2 = df2.values
        self.orginal_dataset2 = df2_org.values
        # print self.discretized_dataset1
        self.outName1 = list(df1.index)
        self.outName2 = list(df2.index)
        # print self.outName1
        # self.outType1 = int
        # self.outType2 = int

        # self.outHead1 = df1.columns
        # self.outHead2 = df2.columns
        # print self.outHead1
        # print df2
        try:
            print("--- %d features and %d samples are used from first dataset" % (
            l1_after, len(self.discretized_dataset1[0])))
        except IndexError:
            sys.exit("WARNING! No feature in the first dataset after filtering.")
        try:
            print("--- %d features and %d samples are used from second dataset" % (
            l2_after, len(self.discretized_dataset2[0])))
        except IndexError:
            sys.exit("WARNING! No feature in the second dataset after filtering.")

        assert (len(self.discretized_dataset1[0]) == len(self.discretized_dataset2[0]))

    def _transform_data(self):
        scale = config.transform_method
        # print(self.orginal_dataset1)
        self.orginal_dataset1 = stats.scale_data(self.orginal_dataset1, scale=scale)
        self.orginal_dataset2 = stats.scale_data(self.orginal_dataset2, scale=scale)
    # print(self.orginal_dataset1)

######logging files
#!/usr/bin/env python

import logging
import pylab
import sys

c_logrbtest = logging.getLogger("btest")


def _log_btest_histograms(aadValues, pFigure, strTitle=None, astrLegend=None):
    c_iN = 20

    figr = pylab.figure() if ((not pFigure) or isinstance(pFigure, str)) else pFigure
    for iValues, adValues in enumerate(aadValues):
        iN = min(c_iN, int((len(adValues) / 5.0) + 0.5))
        iBins, adBins, pPatches = pylab.hist(adValues, iN, normed=1, histtype="stepfilled")
        pylab.setp(pPatches, alpha=0.5)
    if strTitle:
        pylab.title(strTitle)
    if astrLegend:
        pylab.legend(astrLegend, loc="upper left")
    if isinstance(pFigure, str):
        pylab.savefig(pFigure)


def _log_btest_scatter(adX, adY, pFigure, strTitle=None, strX=None, strY=None):
    figr = pylab.figure() if ((not pFigure) or isinstance(pFigure, str)) else pFigure
    pylab.scatter(adX, adY)
    if strTitle:
        pylab.title(strTitle)
    if strX:
        pylab.xlabel(strX)
    if strY:
        pylab.ylabel(strY)
    if isinstance(pFigure, str):
        pylab.savefig(pFigure)


def write_table(data=None, name=None, rowheader=None, colheader=None, prefix="label", col_prefix=None, corner=None,
                delimiter='\t'):
    '''
    wite a matrix of data in tab-delimated format file

    input:
    data: a 2 dimensioal array of data
    name: includes path and the name of file to save
    rowheader
    columnheader

    output:
    a file tabdelimated file

    '''
    if data is None:
        print("Null input for writing table")
        return
    f = open(name, 'w')
    # row numbers as header
    if colheader is None:
        if corner is None:
            f.write(delimiter)
        else:
            f.write(corner)
            f.write(delimiter)
        if col_prefix is None:
            col_prefix = 'S'
        for i in range(len(data[0])):
            f.write(col_prefix + str(i))
            if i < len(data[0]) - 1:
                f.write(delimiter)
        f.write('\n')
    elif len(colheader) == len(data[0]):
        if corner is None:
            f.write(delimiter)
        else:
            f.write(corner)
            f.write(delimiter)
        for i in range(len(data[0])):
            f.write(colheader[i])
            if i < len(data[0]) - 1:
                f.write(delimiter)
        f.write('\n')
    else:
        print("The label list in not matched with the data size")
        sys.exit()

    for i in range(len(data)):
        if rowheader is None:
            f.write(prefix + str(i))
            f.write(delimiter)
        else:
            f.write(rowheader[i])
            f.write(delimiter)
        for j in range(len(data[i])):
            f.write(str(data[i][j]))
            if j < len(data[i]) - 1:
                f.write(delimiter)
        f.write('\n')
    f.close()


def write_circos_table(data, name=None, rowheader=None, colheader=None, prefix="label", corner=None, delimiter='\t'):
    '''
    wite a matrix of data in tab-delimated format file

    input:
    data: a 2 dimensioal array of data
    name: includes path and the name of file to save
    rowheader
    columnheader

    output:
    a file tabdelimated file

    '''
    f = open(name, 'w')

    # write order header
    f.write("Data")
    f.write(delimiter)
    f.write("Data")
    f.write(delimiter)
    for i in range(len(data[0])):
        f.write(str(i + 1))
        if i < len(data[0]) - 1:
            f.write(delimiter)
    f.write('\n')
    # column numbers as header
    f.write("Data")
    f.write(delimiter)
    if len(colheader) == 0:
        f.write("Data")
        f.write(delimiter)
        for i in range(len(data[0])):
            f.write(str(i))
            if i < len(data[0]) - 1:
                f.write(delimiter)
        f.write('\n')
    elif len(colheader) == len(data[0]):
        f.write("Data")
        f.write(delimiter)
        for i in range(len(data[0][:])):
            f.write(colheader[i])
            if i < len(data[0]) - 1:
                f.write(delimiter)
        f.write('\n')
    else:
        sys.err("The lable list in not matched with the data size")
        sys.exit()

    for i in range(len(data)):
        if len(rowheader) == 0:
            f.write(str(i + len(data[0])))
            f.write(prefix + str(i))
            f.write(delimiter)
        elif len(colheader) == len(data[0]):
            f.write(str(i + len(data[0]) + 1))
            f.write(delimiter)
            f.write(rowheader[i])
            f.write(delimiter)
        else:
            sys.err("The lable list in not matched with the data size")
            sys.exit()

        for j in range(len(data[i])):
            f.write(str(data[i][j]))
            if j < len(data[i]) - 1:
                f.write(delimiter)
        f.write('\n')
    f.close()
###############################HSIC
import math
import random
import time

import numpy as np
import scipy

from numpy.random import RandomState
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.stats import gamma, norm

SIGMAS_HSIC = [x for x in range(35000, 85000, 5000)]


def kernelMatrixGaussian(m, m2, sigma=None):
    """
    Calculates kernel matrix with a Gaussian kernel

    m: rows are data points
    m2: rows are data points
    sigma: the bandwidth of the Gaussian kernel.
           If not provided, the median distance between points will be used.

    """

    pairwise_distances = cdist(m, m2, 'sqeuclidean')

    # If sigma is not provided, set sigma based on median distance heuristic.
    if sigma is None:
        sigma = math.sqrt(0.5 * np.median(pairwise_distances[pairwise_distances > 0]))
    gamma = -1.0 / (2 * sigma ** 2)
    return np.exp(gamma * pairwise_distances)


def columnDistanceGaussian(col1, col2, sigma):
    gamma = -1.0 / (2 * sigma ** 2)

    result = np.array([scipy.spatial.distance.sqeuclidean(x, y) for x, y in zip(col1, col2)])
    return np.exp(gamma * np.array(result))


def columnDistanceLinear(col1, col2):
    return np.array([np.dot(x, y) for x, y in zip(col1, col2)])


def getSigmaGaussian(m, m2, sample_size=200, sigma_multiply=0):
    """ Calculate sigma for a gaussian kernel based on observations.

    m: rows are data points
    m2: rows are data points
    sample_size: maximum number of observations to take into account.
                 If number of observation is larger, take random sample
    """
    if m.shape[0] > sample_size:
        prng = RandomState(m.shape[0])  # To have the same bandwidth for the same samples
        ind = prng.choice(m.shape[0], sample_size)
        m = m[ind]
        m2 = m2[ind]
    pairwise_distances = cdist(m, m2, 'sqeuclidean')

    distance_result = np.median(pairwise_distances[pairwise_distances > 0])
    if sigma_multiply != 0:
        distance_result += sigma_multiply * np.std(pairwise_distances[pairwise_distances > 0])
    return math.sqrt(0.5 * distance_result)


def kernelMatrixLinear(m, m2):
    """ Calculates kernel matrix with a linear kernel

    m: rows are data points
    m2: rows are data points
    """
    return np.dot(m, m2.T)


def kernelMatrixDelta(m, m2):
    """ 1 if items are the same. 0 otherwise """
    return 1 - cdist(m, m2, 'hamming')


def columnDistanceDelta(col1, col2):
    return np.array([1 if x == y else 0 for x, y in zip(col1, col2)])


def HSIC_pval_full_gram(X, Y, N_samp=100, kernelX="Gaussian", kernelY="Gaussian", sigmaX=None, sigmaY=None):
    """ Calculates HSIC and p-value

    Old implementation that calculates complete Gramm matrices

    X: Data. Each row is a datapoint.
    Y: Data. Each row is a datapoint.
    N_samp: Number of samples
    kernelX: Kernel to use (Gaussian or Linear)
    kernelY: Kernel to use (Gaussian or Linear)
    """
    timeA = time.time()
    m, _ = X.shape

    # Calculate Gram matrices
    sigmaX = getSigmaGaussian(X, X, 200) if sigmaX is None else sigmaX
    sigmaY = getSigmaGaussian(Y, Y, 200) if sigmaY is None else sigmaY
    K = kernelMatrixGaussian(X, X, sigmaX) if kernelX == "Gaussian" else kernelMatrixLinear(X, X)
    L = kernelMatrixGaussian(Y, Y, sigmaY) if kernelY == "Gaussian" else kernelMatrixLinear(Y, Y)

    # Centering matrix
    H = np.identity(m) - 1.0 / m * np.ones((m, m))
    Kc = np.mat(H) * np.mat(K) * np.mat(H)

    # Dividing by m here, although some papers use m-1
    HSIC = np.trace(np.dot(np.transpose(Kc), L)) / m ** 2

    boots = []
    Yrand = np.copy(Y)
    for _ in xrange(N_samp):
        np.random.shuffle(Yrand)
        L = kernelMatrixGaussian(Yrand, Yrand) if kernelY == "Gaussian" else kernelMatrixLinear(Yrand, Yrand)
        boots.append(np.trace(np.dot(np.transpose(Kc), L)) / m ** 2)

    boots = np.array(boots)
    pval = (sum(b >= HSIC for b in boots) + 1) / float(len(boots) + 1)
    return HSIC, pval


def HSIC_pval(X, Y, N_samp=500, kernelX="Gaussian", kernelY="Gaussian", eta=0.001,
              sigmaX=None, sigmaY=None,
              p_method="boots", return_boots=False):
    """ Calculates HSIC and p-value

    Gram matrices are approximated using incomplete Cholesky decomposition.

    X: Data. Each row is a datapoint.
    Y: Data. Each row is a datapoint.
    N_samp: Number of samples
    kernelX: Kernel to use (Gaussian, Linear, Delta)
    kernelY: Kernel to use (Gaussian, Linear, Delta)
    eta: Threshold for incomplete Cholesky decomposition
    sigmaX: sigma for X when using Gaussian kernel
    sigmaY: sigma for Y when using Gaussian kernel
    """
    timeA = time.time()
    m, _ = X.shape

    sigmaX = getSigmaGaussian(X, X, 200) if sigmaX is None else sigmaX
    sigmaY = getSigmaGaussian(Y, Y, 200) if sigmaY is None else sigmaY

    A, max_rankA = incompleteCholeskyKernel(X, m, kernelX, sigmaX, eta)
    B, max_rankB = incompleteCholeskyKernel(Y, m, kernelY, sigmaY, eta)

    centered_A = A.T - A.T.mean(axis=0)
    tmp = B * np.mat(centered_A)
    HSIC = np.trace(tmp * tmp.T) / m ** 2

    boots = []
    Yrand = np.copy(Y)
    for _ in xrange(N_samp):
        np.random.shuffle(Yrand)

        B, max_rankB = incompleteCholeskyKernel(Yrand, m, kernelY, sigmaY, eta)

        tmp = np.mat(B) * np.mat(centered_A)
        boots.append(np.trace(tmp * tmp.T) / m ** 2)

    boots = np.array(boots)

    if p_method == "boots":
        pval = (sum(b >= HSIC for b in boots) + 1) / float(len(boots) + 1)
    else:  # gamma
        fit_alpha, fit_loc, fit_beta = gamma.fit(boots)
        pval = 1 - gamma.cdf(HSIC, fit_alpha, scale=fit_beta, loc=fit_loc)
    if return_boots:
        return HSIC, pval, boots
    else:
        return HSIC, pval


def HSIC_pval_bandwidth_sweep(locs, has_word, N_samp=500, kernelX="Gaussian", kernelY="Gaussian", eta=0.001):
    """" Calculate HSIC by sweeping over bandwidth values """
    HSIC_vals = []
    HSIC_pvals = []
    best_HSIC_val = None
    best_pval = float("inf")
    for s in SIGMAS_HSIC:
        val, pval = HSIC_pval(locs, has_word, N_samp, kernelX, kernelY, eta, sigmaX=s)
        HSIC_vals.append(val)
        HSIC_pvals.append(pval)

        if pval < best_pval:
            best_pval = pval
            best_HSIC_val = val

    return best_HSIC_val, best_pval, HSIC_vals, HSIC_pvals


def incompleteCholesky(K, k, eta=0.01):
    """ Incomplete Cholesky decomposition
    Based on algorithm in Kernel Methods for Pattern Analysis, chapter
    Elementary algorithms in feature space, fragment 5.4

    K: the matrix
    k: numbers of rows for new matrix
    eta: threshold
    """
    ell, _ = K.shape
    I = []
    R = np.zeros((ell, ell))
    d = np.diagonal(K).copy()
    a = max(d)
    I.append(np.argmax(d))
    j = 0
    while a > eta and j < k:
        nu_j = math.sqrt(a)
        for i in xrange(ell):
            R[j, i] = (K[I[j], i] - np.dot(R[:, i].T, R[:, I[j]])) / nu_j
        d = d - R[j, :] ** 2
        a = max(d)
        I.append(np.argmax(d))
        j += 1

    return R[:j, ], j


def incompleteCholeskyKernel(X, maxrank, kernel, sigma=None, eta=0.001):
    """ Incomplete Cholesky decomposition
    Based on algorithm in Kernel Methods for Pattern Analysis, chapter
    Elementary algorithms in feature space, fragment 5.4.
    Doesn't need to compute Gram matrix beforehand.

    K: the matrix
    k: numbers of rows for new matrix
    kernel: kernel to use
    sigma: in case of Gaussian kernel
    eta: threshold
    """
    maxrank = min(maxrank, 100)
    ell, _ = X.shape
    I = []
    R = np.zeros((maxrank, ell))

    d = None
    if kernel == "Gaussian":
        d = columnDistanceGaussian(X, X, sigma)
    elif kernel == "Linear":
        d = columnDistanceLinear(X, X)
    elif kernel == "Delta":
        d = columnDistanceDelta(X, X)

    a = max(d)
    I.append(np.argmax(d))
    j = 0
    while j < maxrank and a > eta:
        nu_j = math.sqrt(a)
        x_elem = np.atleast_2d(X[I[j]])

        K_tmp = None
        if kernel == "Gaussian":
            K_tmp = kernelMatrixGaussian(x_elem, X, sigma)
        elif kernel == "Linear":
            K_tmp = kernelMatrixLinear(x_elem, X)
        elif kernel == "Delta":
            K_tmp = kernelMatrixDelta(x_elem, X)

        for i in xrange(ell):
            R[j, i] = (K_tmp[0][i] - np.dot(R[:, i].T, R[:, I[j]])) / nu_j
        d = d - R[j, :] ** 2
        a = max(d)
        I.append(np.argmax(d))
        j += 1

    return R[:j, ], j
