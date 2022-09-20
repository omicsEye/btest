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
from . import utils
from . import config
from . import blockplot


def get_btest_base_directory():
    """
    Return the location of the btest base directory
    """

    config_file_location = os.path.dirname(os.path.realpath(__file__))

    # The btest base directory is parent directory of the config file location
    btest_base_directory = os.path.abspath(os.path.join(config_file_location, os.pardir))

    return btest_base_directory


def check_requirements(args):
    """
    Check requirements (file format, dependencies, permissions)
    """
    # check the third party softwares for plotting the results
    try:
        import pandas as pd
    except ImportError:
        sys.exit("--- Please check your installation for pandas library")
    # Check that the output directory is writeable
    output_dir = os.path.abspath(args.output_dir)
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

def set_parameters(args):
    '''
    Set the user command line options to config file
    to be used in the program
    '''
    config.output_dir = args.output_dir

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
        "-m",
        dest="strMetric",
        default='spearman',
        choices=[ "pearson", "spearman", "kendall"],
        help="metric to be used for similarity measurement\n[default = 'spearman']")

    argp.add_argument(
        "--fdr",
        dest="fdr",
        type=float,
        default=0.1,
        help="Target FDR correction using BH approach")
    argp.add_argument(
        "--var",
        dest="min_var",
        type=float,
        default=0.0,
        help="Minimum variation to keep a feature in tests")

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

    return argp.parse_args()


def btest(X_path, Y_path,
          outputpath,
          method='spearman',
          plot=True,
          fdr=0.1,
          min_var=0.0
          ):
    # set the parameter to config file
    dataX , dataY = utils.readData(X_path, Y_path)
    dataAll, featuresX, featuresY = utils.dataProcess(dataX, dataY, min_var=min_var)
    within_X, within_Y, X_Y, rho_X, rho_Y, rho_X_Y = utils.corr_paired_data(dataAll, featuresX, featuresY, method=method, fdr=fdr)
    utils.write_results(within_X, within_Y, X_Y, rho_X, rho_Y, rho_X_Y, outputpath)
    if plot:
        associations = blockplot.load_associations(path=outputpath + '/X_Y.tsv')
        simtable = blockplot.load_order_table(outputpath + '/simtable.tsv', associations)
        blockplot.plot(
            simtable,
            associations,
            cmap="RdBu_r",
            mask=False,
            axlabels=["", ""],
            outfile=outputpath + "/blockplot.pdf",
            similarity="Spearman"
        )

    return within_X, within_Y, X_Y, rho_X, rho_Y, rho_X_Y

def main():
    # Parse arguments from command line
    args = parse_arguments(sys.argv)

    # set the parameter to config file
    set_parameters(args)

    # check the requiremnts based on need for parameters
    check_requirements(args)

    # run btest approach
    results = btest(X_path=args.X, Y_path=args.Y, outputpath=args.output_dir, method=args.strMetric, fdr=args.fdr, min_var=args.min_var)


if __name__ == '__main__':
    main()

