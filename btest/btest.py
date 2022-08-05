#!/usr/bin/env python
# user options handling plus check the requirements
import sys

# Check the python version
REQUIRED_PYTHON_VERSION_MAJOR = [2, 3]
REQUIRED_PYTHON_VERSION_MINOR2 = 7
REQUIRED_PYTHON_VERSION_MINOR3 = 6
try:
    if (not sys.version_info[0] in REQUIRED_PYTHON_VERSION_MAJOR or
            (sys.version_info[0] == 2 and sys.version_info[1] < REQUIRED_PYTHON_VERSION_MINOR2) or
            (sys.version_info[0] == 3 and sys.version_info[1] < REQUIRED_PYTHON_VERSION_MINOR3)):
        sys.exit("CRITICAL ERROR: The python version found (version " +
                 str(sys.version_info[0]) + "." + str(sys.version_info[1]) + ") " +
                 "does not match the version required (version " +
                 str(2) + "." +
                 str(REQUIRED_PYTHON_VERSION_MINOR2) + "+) or " +
                 str(3) + "." +
                 str(REQUIRED_PYTHON_VERSION_MINOR3) + "+)")
except (AttributeError, IndexError):
    sys.exit("CRITICAL ERROR: The python version found (version 1) " +
             "does not match the version required (version " +
             str(REQUIRED_PYTHON_VERSION_MAJOR) + "." +
             str(REQUIRED_PYTHON_VERSION_MINOR) + "+)")

import argparse
import csv
import itertools
import logging
import os
import shutil
import time
import math
import random
from scipy.stats import pearsonr
import numpy as np

# Test if numpy is installed
try:
    from numpy import array
    import numpy as np
except ImportError:
    sys.exit("Please install numpy")

# Test if matplotlib is installed
try:
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except ImportError:
    sys.exit("Please install matplotlib")

#  Load a btest module to check the installation
# try:
from . import store
# except ImportError:
#   sys.exit("CRITICAL ERROR: Unable to find the store module." +
#       " Please check your btest install.")

from . import utilities
from . import stats
from . import config


def get_btest_base_directory():
    """
    Return the location of the btest base directory
    """

    config_file_location = os.path.dirname(os.path.realpath(__file__))

    # The btest base directory is parent directory of the config file location
    btest_base_directory = os.path.abspath(os.path.join(config_file_location, os.pardir))

    return btest_base_directory


def check_requirements():
    """
    Check requirements (file format, dependencies, permissions)
    """
    # check the third party softwares for plotting the results
    try:
        import pandas as pd
    except ImportError:
        sys.exit("--- Please check your installation for pandas library")
    # Check that the output directory is writeable
    output_dir = os.path.abspath(config.output_dir)
    if not os.path.isdir(output_dir):
        try:
            print("Creating output directory: " + output_dir)
            os.mkdir(output_dir)
        except EnvironmentError:
            sys.exit("CRITICAL ERROR: Unable to create output directory.")
    '''else:
        try:
            print("will rewrite files as the output directory exist" + output_dir)
            shutil.rmtree(output_dir)
            print("Creating output directory: " + output_dir)
            os.mkdir(output_dir)
        except EnvironmentError:
            sys.exit("CRITICAL ERROR2: Unable to create output directory.")
        '''

    if not os.access(output_dir, os.W_OK):
        sys.exit("CRITICAL ERROR: The output directory is not " +
                 "writeable. This software needs to write files to this directory.\n" +
                 "Please select another directory.")

    print("Output files will be written to: " + output_dir)
    if config.similarity_method == 'mic':
        try:
            import minepy
        except ImportError:
            sys.exit("--- Please check minepy installation for MIC library")
    # if mca is chosen as decomposition method check if it's R package and dependencies are installed
    if config.decomposition == 'mca':
        try:
            from rpy2 import robjects as ro
            from rpy2.robjects import r
            from rpy2.robjects.packages import importr
            import rpy2.robjects.numpy2ri
            rpy2.robjects.numpy2ri.activate()
            ro.r('library(FactoMineR)')
        except:
            sys.exit("--- Please check R, rpy2,  and  FactoMineR installation for MCA library")


def parse_arguments(args):
    """
    Parse the arguments from the user
    """
    argp = argparse.ArgumentParser(
        description="block-wise association testing",
        formatter_class=argparse.RawTextHelpFormatter,
        prog="btest")
    argp.add_argument(
        "--version",
        action="version",
        version="%(prog)s " + config.version)
    argp.add_argument(
        "-X", metavar="<input_dataset_1.txt>",
        type=argparse.FileType("r"), default=sys.stdin,
        help="first file: Tab-delimited text input file, one row per feature, one column per measurement\n[REQUIRED]",
        required=True)

    argp.add_argument(
        "-Y", metavar="<input_dataset_2.txt>",
        type=argparse.FileType("r"),
        default=None,
        help="second file: Tab-delimited text input file, one row per feature, one column per measurement\n[default = the first file (-X)]")

    argp.add_argument(
        "-o", "--output",
        dest="output_dir",
        help="directory to write output files\n[REQUIRED]",
        metavar="<output>",
        required=True)

    argp.add_argument(
        "-v", "--verbose",
        dest="verbose",
        default=config.verbose,
        help="additional output is printed")

    argp.add_argument(
        "--diagnostics-plot",
        dest="diagnostics_plot",
        help="Diagnostics plot for associations ",
        action="store_true")
    argp.add_argument(
        "--header",
        action="store_true",
        help="the input files contain a header line")
    argp.add_argument(
        "--format-feature-names",
        dest="format_feature_names",
        help="Replaces special characters and for OTUs separated  by | uses the known end of a clade",
        action="store_true")
    argp.add_argument(
        "-s", "--seed",
        type=int,
        default=0,  # random.randint(1,10000),
        help="a seed number to make the random permutation reproducible\n[default = 0,and -1 for random number]")
    argp.add_argument(
        "-e", "--entropy",
        dest="entropy_threshold",
        type=float,
        default=0.5,
        help="Minimum entropy threshold to filter features with low information\n[default = 0.5]")
    argp.add_argument(
        "-e1", "--entropy1",
        dest="entropy_threshold1",
        type=float,
        default=None,
        help="Minimum entropy threshold for the first dataset \n[default = None]")
    argp.add_argument(
        "-e2", "--entropy2",
        dest="entropy_threshold2",
        type=float,
        default=None,
        help="Minimum entropy threshold for the second dataset \n[default = None]")
    argp.add_argument(
        "--missing-char",
        dest="missing_char",
        default='',
        help="defines missing characters\n[default = '']")
    argp.add_argument(
        "--fill-missing",
        dest="missing_method",
        default=None,
        choices=["mean", "median", "most_frequent"],
        help="defines missing strategy to fill missing data.\nFor categorical data puts all missing data in one new category.")
    return argp.parse_args()


def btest(X, Y,
            output_dir='.',
            e1=None, e2=None,
            e=0.5,
            missing_char='',
            missing_method=None,
            i=1000,
            linkage_method='average',
            discretizing='equal-freq',
            m='',
            btestgram=True, \
            diagnostics_plot=True, \
            header=False, format_feature_names=False, \
            nproc=1, nbin=None, s=0, \
            missing_char_category=False,
            write_hypothesis_tree=False, t=''):
    # set the paramater to config file
    config.similarity_method = m.lower()
    config.entropy_threshold = e
    if e1 == None:
        config.entropy_threshold1 = e
    else:
        config.entropy_threshold1 = e1
    if e2 == None:
        config.entropy_threshold2 = e
    else:
        config.entropy_threshold2 = e2
    # otherwise use gpd as fast and accurate pvalue calculation approach
    config.missing_char = missing_char
    config.missing_method = missing_method
    config.missing_char_category = missing_char_category
    # load_input()

    # read  input files
    dataX = pd.read_table(dataX, index_col=0, header=0)
    # print(data.shape)
    # print(data.index)
    # print(data.columns)

    dataY = pd.read_table(dataY, index_col=0, header=0)
    #   print(data.index)
    # print(metadata.index)
    ind = dataY.index.intersection(dataX.index)

    if len(ind) != data.shape[0]:
        print("the data and metadata have different number of rows and number of common rows is: ", len(ind))
        print("The number of missing metadata are: ", data.shape[0] - len(ind))
        diff_rows = data.index.difference(metadata.index)
        # print (diff_rows)
        empty_section_metadata = pd.DataFrame(index=diff_rows, columns=metadata.columns)
        metadata = pd.concat([metadata, empty_section_metadata])
    dataY = dataY.loc[dataX.index, :]
    check_requirements()
    doTest(dataX, dataY)
    return
def doTest(dataX, dataY):

    df = pd.concat([dataX, dataY])
    rho = df.corr()
    pval = df.corr(method=lambda x, y: pearsonr(x, y)[1]) - np.eye(*rho.shape)
    rho = pd.melt(rho)
    pval = pd.melt(pval)
    result = pd.concat([df1, df3], axis=1)
def main():
    # Parse arguments from command line
    args = parse_arguments(sys.argv)

    # set the parameter to config file
    set_parameters(args)

    # check the requiremnts based on need for parameters
    check_requirements()

    # run btest approach
    doTest(dataX, dataY)


if __name__ == '__main__':
    main()

