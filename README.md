# btest #

btest: a method to link, rank, and visualize associations among omics features across multi-omics datasets
The tool is for general purpose,  and well-powered association discovery in paired multi-omic datasets. 

----

**Citation:**

Bahar Sayoldin, Keith A. Crandall,  Ali Rahnavard, *btest: link, rank, and visualize associations among omics features across multi-omics datasets* [https://rahnavard.org/btest](https://rahnavard.org/btest)

* For installation and a quick demo, read the [Initial Installation](#initial-installation)


_`btest`_ combines block nonparametric hypothesis testing with false discovery rate correction to 
enable high-sensitivity discovery of linear and non-linear associations in high-dimensional datasets 
(which may be categorical, continuous, or mixed). btest perform test by: 
1) adjust data to covariates, 
2) hierarchically clustering to pair block of significantly associated features, 
3) provide informative association by incorporating within data sets and among data sets relationships, and
4) high quality visualization of data. 

---
Please join the community discussion forum at [omicsEye/btest](https://forum.omicseye.org/)  

## Contents ##

* [Features](#features)
* [Overview workflow](#overview-workflow)
* [Requirements](#requirements)
    * [Software](#software)
    * [Other](#other)
* [Initial Installation](#initial-installation)
    1. [Install btest](#1-install-btest)
    2. [Test the install](#2-test-the-install)
    3. [Try out a demo run](#3-try-out-a-demo-run)
* [How to run](#how-to-run)
    * [Basic usage](#basic-usage)
    * [Setting for association types](#setting-for-association-types)
    * [Demo runs](#demo-runs)
* [Output files](#output-files)
    1. [Associations file](#1-associations-file)
    2. [Similarity table](#2-similarity-table)
    3. [Hypothesis tree](#3-hypothesis-tree)
* [Result plots](#result-plots)
    1. [First dataset heatmap](#1-first-dataset-heatmap)
    2. [Second dataset heatmap](#2-second-dataset-heatmap)
    3. [Associations blockplot](#3-associations-blockplot)
    4. [Diagnostics scatter or confusion matrix plot](#4-diagnostics-scatter-or-confusion-matrix-plot)
* [Configuration](#configuration)
* [Tutorials](#tutorials)
    * [Adjust for covariates](#adjust-for-covariates)
    * [Selecting a desired false discovery rate](#selecting-a-desired-false-discovery-rate)
    * [Selecting a similarity measurement ](#selecting-a-similarity-measurement)
    * [Selecting a decomposition method ](#selecting-a-decomposition-method)
    * [Pairwise association testsing or AllA](#pairwise-association-testsing-or-alla)
    * [Filtering features from input by minimum entropy](#filtering-features-from-input-by-minimum-entropy)
* [Tools](#tools)
    * [datasim: synthetic paired dataset generator](#datasim-a-tool-for-synthetic-data)
    * [blockplot](#blockplot-a-tool-for-visualization)
    * [scatter](#scatter-a-tool-for-visualization)
    * [btest Python API](#btest-python-api)
    * [Nonparametric p value](#nonparametric-p-value)
* [FAQs](#faqs)
* [Complete option list](#complete-option-list)

## Features ##

1. Generality: btest can handle datasets from various omics profiles

2. Efficiency: design and implementation of function tend to work with large data

3. Reliability: btest utilizes multiple hypothesis testing in paired omics data.

4. False discovery rate correction (FDR) using Benjamini–Hochberg(BH).

6. A simple user interface (single command driven flow)

    * The user only needs to provide a paired dataset


## Overview workflow ##

![](http://github.co/omicsEye/btest/img/fig1.png)

*   File Types: tab-delimited text file with columns headers as samples or no sample names with the same order samples without head and row names as features
 
## Requirements ##

### Software ###

1. [Python](http://www.python.org/) (version >= 2.7 or >= 3.4)
2. [Numpy](http://http://www.numpy.org/) (version >= 1.9.2) (automatically installed)
3. [Scipy](https://www.scipy.org/) (version >= 0.17.1) (automatically installed)
4. [Matplotlib](http://http://matplotlib.org/) (version >= 1.5.1) (automatically installed)
5. [Scikit-learn](http://http://scikit-learn.org/stable/) (version >= 0.14.1) (automatically installed)
6. [pandas](https://http://pandas.pydata.org/pandas-docs/stable/) (version >= 0.18.1) (automatically installed)

### Other ###

1. Memory depends on input size mainly the number of features in each dataset 
2. Runtime depends on input size mainly the number of features in each dataset and similarity score that has been chosen
3. Operating system (Linux, Mac, or Windows)

## Initial Installation ##


### 1. Install btest ###

#### Installing from source ####

## INSTALLATION ##
* First install *conda*  
Go to the [Anaconda website](https://www.anaconda.com/) and download the latest version for your operating system.  
* For Windows users: DO NOT FORGET TO ADD CONDA TO your system PATH*
* Second is to check for conda availability  
open a terminal (or command line for Windows users) and run:
```
conda --version
```
it should out put something like:
```
conda 4.9.2
```
<span style="color:#fc0335">if not, you must make *conda* available to your system for further steps.</span>
if you have problems adding conda to PATH, you can find instructions [here](https://docs.anaconda.com/anaconda/user-guide/faq/).  

### Windows Linux Mac ###
If you are **NOT** using an **Apple M1 MAC** please go to the [Apple M1 MAC](#apple-m1-mac) for installation instructions.  
<span style="color:#033C5A">*If you have a working conda on your system, you can safely skip to step three*</span>.
1) Create a new conda environment (let's call it btest_env) with the following command:
```
conda create --name btest_env python=3.8
```
2) Activate your conda environment:
```commandline
conda activate btest_env 
```
3) Install *btest*:
you can directly install if from GitHub:
```commandline
python -m pip install git+https://github.com/omicsEye/btest
```
### Apple M1 MAC ###
1) Update/install Xcode Command Line Tools
  ```commandline
  xcode-select --install
  ```
2) Install [Brew](https://brew.sh/index_fr)
  ```commandline
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
  ```
3) Install libraries for brew
  ```commandline
  brew install cmake libomp
  ```
4) Install miniforge
  ```commandline
  brew install miniforge
  ```
5) Close the current terminal and open a new terminal
6) Create a new conda environment (let's call it btest_env) with the following command:
  ```commandline
  conda create --name btest_env python=3.8
  ```
7) Activate the conda environment
  ```commandline
  conda activate btest_env
  ```
8) Install packages from Conda
  ```commandline
  conda install numpy scipy scikit-learn==0.23.2
  ```
  Then
  ```commandline
  conda install lightgbm
  pip install xgboost
  ```
9) Finally, install *btest*:
you can directly install if from GitHub:
```commandline
python -m pip install git+https://github.com/omicsEye/btest
```

#### Installing with pip ####

* $ pip install btest ``
* This command will automatically install btest and its dependencies.
* To overwrite existing installations of dependencies use "-U" to force update them. 
* To use the existing version of dependencies use "--no-dependencies." 
* If you do not have write permissions to '/usr/lib/,' then add the option "--user" to the btest install command. Using this option will install the python package into subdirectories of '~/.local' on Linux. Please note when using the "--user" install option on some platforms, you might need to add '~/.local/bin/' to your $PATH as default might not include it. You will know if it needs to be added if you see the following message `btest: command not found` when trying to run btest after installing with the "--user" option.
* If you use Windows operating system you can install it with administrator permission easily (please open a terminal with administrator permission and the rest is the same). 


------------------------------------------------------------------------------------------------------------------------------

# Getting Started with btest #

## Test btest ##

To test if btest is installed correctly, you may run the following command in the terminal:

```#!cmd
btest -h
```
Which yields btest command line options.
```commandline
usage: btest [-h] [--version] -X <input_dataset_1.txt> [-Y <input_dataset_2.txt>] -o <output> [-m {pearson,spearman,kendall}] [--fdr FDR] [--var MIN_VAR] [-v VERBOSE] [--diagnostics-plot]
             [--header] [--format-feature-names] [-s SEED]

block-wise association testing

optional arguments:
  -h, --help            show this help message and exit
  --version             show program's version number and exit
  -X <input_dataset_1.txt>
                        first file: Tab-delimited text input file, one row per feature, one column per measurement
                        [REQUIRED]
  -Y <input_dataset_2.txt>
                        second file: Tab-delimited text input file, one row per feature, one column per measurement
                        [default = the first file (-X)]
  -o <output>, --output <output>
                        directory to write output files
                        [REQUIRED]
  -m {pearson,spearman,kendall}
                        metric to be used for similarity measurement
                        [default = 'spearman']
  --fdr FDR             Target FDR correction using BH approach
  --var MIN_VAR         Minimum variation to keep a feature in tests
  -v VERBOSE, --verbose VERBOSE
                        additional output is printed
  --diagnostics-plot    Diagnostics plot for associations 
  --header              the input files contain a header line
  --format-feature-names
                        Replaces special characters and for OTUs separated  by | uses the known end of a clade
  -s SEED, --seed SEED  a seed number to make the random permutation reproducible
                        [default = 0,and -1 for random number]
```
### 2. Test the installation ###

1. Test out the install with unit and functional tests
     * `` $ btest_test``

### 3. Try out a demo run ###


With btest installed you can try out a demo run using two sample synthetic datasets. 
``$ btest -X demo/X_dataset.txt -Y demo/Y_dataset.txt -o $OUTPUT_DIR --fdr 0.1``

The output from this demo run will be written to the folder $OUTPUT_DIR.

## Installation Update ##

If you have already installed btest, using the [Initial Installation](#initial-installation) steps, and would like to upgrade your installed version to the latest version, please do:

`sudo -H pip install btest --upgrade --no-deps` or
`pip install btest --upgrade --no-deps`

This command upgrades btest to the latest version and ignores updating btest's dependencies. 



## Output files ##

When btest is completed, three main output files will be created:

### 1. Associations file ###

``` 
| association_rank | cluster1                | cluster1_similarity_score | cluster2                | cluster2_similarity_score | pvalue   | qvalue      | similarity_score_between_clusters |
|------------------|-------------------------|---------------------------|-------------------------|---------------------------|----------|-------------|-----------------------------------|
| 1                | X30;X31                 | 0.738949895               | Y30;Y31                 | 0.562388239               | 3.33E-37 | 2.81E-34    | -0.900426043                      |
| 2                | X7;X10;X11;X9;X6;X8     | 0.521149715               | Y7;Y10;Y11;Y8;Y6;Y9     | 0.478449445               | 6.91E-32 | 2.92E-29    | -0.870183018                      |
| 3                | X16;X17;X15;X13;X12;X14 | 0.466724272               | Y16;Y13;Y17;Y15;Y12;Y14 | 0.400633663               | 2.94E-31 | 8.28E-29    | -0.866006601                      |
| 4                | X1;X3;X2;X4;X0;X5       | 0.567457546               | Y3;Y1;Y5;Y2;Y0;Y4       | 0.458731473               | 1.33E-28 | 2.81E-26    | -0.846672667                      |
| 5                | X28;X27;X26;X25;X24;X29 | 0.502168617               | Y27;Y28;Y25;Y26;Y24;Y29 | 0.414425443               | 4.91E-26 | 8.30E-24    | -0.825058506                      |
| 6                | X22;X20;X23;X19;X18;X21 | 0.511786379               | Y20;Y21;Y18;Y22;Y19;Y23 | 0.415246325               | 3.39E-23 | 4.77E-21    | -0.797119712                      |
| 7                | X0;X5                   | 0.781482148               | Y20                     | 1                         | 9.12E-05 | 0.011005714 | 0.381206121                       |
```

*   File name: `` $OUTPUT_DIR/associations.txt ``
*   This file details the associations. Features are grouped in clusters that participated in an association with another cluster.
*    **```association_rank```**: associations are sorted by high similarity score and low p-values.
*    **```cluster1```**: has one or more homogenous features from the first dataset that participate in the association.
*    **```cluster1_similarity_score```**: this value is corresponding to `1 - condensed distance` of the cluster in the hierarchy of the first dataset.
*    **```cluster2```**: has one or more homogenous features from the second dataset that participate in the association.
*    **```cluster2_similarity_score```**: this value is corresponding to `1 - condensed distance` of the cluster in the hierarchy of the second dataset.
*    **```pvalue```**: p-value from Benjamini-Hochbergapproach used to assess the statistical significance of the mutual information distance.
*    **```qvalue```**: q value calculates after BH correction for each test.
*    **```similarity_score_between_clusters```**: is the similarity score of the representatives (medoids) of two clusters in the association.

## Output files ##
1. [First dataset heatmap](#1-first-dataset-heatmap)
    2. [Second dataset heatmap](#2-second-dataset-heatmap)
    3. [Associations blockplot](#3-associations-blockplot)
    4. [Diagnostics scatter or confusion matrix plot](#4-diagnostics-plot)

### 1. First dataset heatmap ###
![](github.com/omicsEye/btest/img/hierarchical_heatmap_spearman_1.png =15x)

### 2. Second dataset heatmap ###
![](github.com/omicsEye/btest/img/hierarchical_heatmap_spearman_2.png =15x)

### 3. Associations blockplot ###
![](github.com/omicsEye/btest/img/blockplot_strongest_7.png =20x)

*   File name: `` $OUTPUT_DIR/blockplot.pdf ``
*   This file has a visualized representative of results in a heatmap. Rows are the features from the first dataset that participated in at least on association and the orders comes from their order in `linkage hierarchical cluster` with `average method`. Columns are the features from the second dataset that participated in at least on association and the orders comes from their order in `linkage hierarchical cluster` with `average method`. 
*   Each cell color represents the pairwise similarity between individual features.
*   Number on each block represents significant association numbers which are based on `similarity score` descending order (largest first) with p-value ascending order in a case of the same `similarity score`.

### 4. Diagnostics scatter or confusion matrix plot ###
![](github.com/omicsEye/btest/img/Scatter_association1.png =20x)

*   If option `` --diagnostics-plot`` is provided with ``btest`` command line then for each association, a set of plots will be produced at the end of btest's process.
*   File name: `` $OUTPUT_DIR/diagnostics_plot/association_1/Scatter_association1.pdf ``
*   This file has a visualized representative of Association 1 in a heatmap. X's are closer of features from a cluster in the first dataset that is significantly associated with a cluster of features, Ys, in the second dataset. The scatter plot shows how the association looks like within cluster and between initial features. 


## Configuration ##

btest produces a performance file to store user configuration settings. This configuration file is automatically created in the output directory.

```sh
$ vi btest.log
btest version:	1.1.1

```

## Tutorials ##

### Adjust for covariates ###

btest can be used to test the relationship between metadata (e.g. age and gender) and data (e.g. microbial species abundance and immune cell counts). In this case, related (covaried) metadata cluster together. In circumstances that two datasets are tested such as microbiome vs. metabolites, the effect of covariates (e.g. age, gender, and batch effect)  from both datasets such as (microbial species and metabolites) should be regressed out. Users should adjust for covariates. Here we provide two examples of R programming that how to adjust for a variable.

* **Adjust for age**: let's regress out the age effect from microbial species or metabolites:
 
```
#!python

lmer(microbe ~ age, microbial_abundance_data = table)
lmer(metabolite ~ age, metabolites_data = table)
```

* **Adjust for time**: this type of adjustment with groups structure involving has more complexity for adjusting and we recommend to read Winkler et al. Neuroimage. 2014 entitled *Permutation inference for the general linear model.* A simple code for this case would be: assume we have microbial samples from the same subject in several time-points a linear mixed-effects model is fit using the R lme4 package to each microbial species or metabolites of the form:


```
#!python

lmer(microbe ~ 1 + (1 | subject) + time, microbial_abundance_data = table)
```


### Selecting a desired false discovery rate ###

btest by default use 0.1 as the target false discovery rate. Users can change it to the desired value, for example, 0.05 or 0.25 by using `-q 0.05`.

### Selecting a similarity measurement ###

btest’s implementation and hypothesis testing scheme are highly general, allowing them to be used with a wide variety of similar measures. For similarity measurement option we recommend: 1) Spearman coefficient for continues data, 2)(default for btest) normalized mutual information (NMI) for mixed data (continuous, categorical, and binary data), and 3) discretized maximum information coefficient (DMIC) for complicated associations types such as sine waves in continuous data.  Similarity measures are implemented in the current version of btest that user can use as options are: Spearman coefficient, discretized normalized mutual information, discretized adjusted mutual information, discretized maximal information coefficient, Pearson correlation, distance correlation (dCor).  
``-m spearman`` for example change the default similarity to Spearman coefficient as similarity measurement, and it automatically bypasses discretizing step. For available similarity metrics, please look at btest options using ``btest -h``.

### Selecting a decomposition method ###

btest uses medoid of clusters as a representative to test the relation between clusters. A user can use other options using ``-d`` with other decomposition methods such as PCA, ICA, MCA. For example, ``-d pca`` will use the first principal component of a cluster as its representative.   

### Pairwise association testsing or AllA ###

A user can choose AllA as a naive pairwise testing approach using ``-a pair`` option in the command line where the default is ``-a block`` which uses the hierarchical approach.

### Filtering features from input by minimum entropy ###

btest by default removes features with low entropy (<.5) to reduce the unnecessary number tests. A user can use different threshold using option ``-e $THRESHOLD``. $THRESHOLD by default is .5.

## Tools ##


### blockplot: a tool for visualization ###

btest includes tools to be used with results.


`` $ cd $OUTPUT_DIR ``

`` $ blockplot  $blockplot similarity_table.txt hypotheses_tree.txt associations.txt blockplot.pdf ``

*   $TABLE = gene/pathway table (tsv or biom format)
*   $OUTPUT_DIR = the directory to write new gene/pathway tables (one per sample, in biom format if input is biom format)
```
usage: blockplot [-h] [--strongest STRONGEST] [--largest LARGEST] [--mask]
                 [--cmap CMAP] [--axlabels AXLABELS AXLABELS]
                 [--outfile OUTFILE] [--similarity SIMILARITY]
                 [--orderby ORDERBY]
                 simtable tree associations

positional arguments:
  simtable              table of pairwise similarity scores
  tree                  hypothesis tree (for getting feature order)
  associations          btest associations

optional arguments:
  -h, --help            show this help message and exit
  --strongest STRONGEST
                        isolate the N strongest associations
  --largest LARGEST     isolate the N largest associations
  --mask                mask feature pairs not in associations
  --cmap CMAP           matplotlib color map
  --axlabels AXLABELS AXLABELS
                        axis labels
  --outfile OUTFILE     output file name
  --similarity SIMILARITY
                        Similarity metric has been used for similarity
                        measurement
  --orderby ORDERBY     Order the significant association by similarity,
                        pvalue, or qvalue
```

### scatter: a tool for visualization ###
btest provides a script `scatter` to make a scatter matrix of between all features participate in an association.

`` $ scatter 1 --input ./ --outfile scatter_1.pdf``

```
usage: scatter [-h] [--input INPUT] [--outfile OUTFILE]
                    association_number

positional arguments:
  association_number  Association number to be plotted

optional arguments:
  -h, --help          show this help message and exit
  --input INPUT       btest output directory
  --outfile OUTFILE   output file name
```

### datasim: a tool for synthetic data ###
datasim generates paired datasets with various properties including: the size (number of features (rows) and samples (columns)),  the number of blocks (clusters within each dataset, the structure of clusters, the type of associations between features,  distribution of data (normal and uniform), the structure of clustering with each dataset,  the strongness of association between cluster among datasets define by noise between associated blocks, and the strongness of similarity between features within clusters defined by noise within blocks.

Here are two examples to generate paired datasets with the associations between them and btest runs. 

`datasim -f 32 -n 100 -a line -d uniform -s balanced -o btest`

The outputs will be located in `btest_data_f32_s100_line` directory and include a paired datasets: `X_line_32_100.txt`    `Y_line_32_100.txt` and `A_line_32_100.txt` association between them. A's rows are features in X dataset, and A's columns are features in Y dataset and for each cell in A zero means no significant association and 1 mean significant association. 
To run btest use on this synthetic data use:

`btest -X btest/X_line_32_100.txt -Y btest/Y_line_32_100.txt -o btest_output_f32_n100_line_spearman`
As all features in datasets are continuous btest uses Spearman coefficient as the similarity metric. One can specify a different similarity metric.  For example, try the same dataset with Normalized Mutual Information:
`btest -X btest/X_line_32_100.txt -Y btest/Y_line_32_100.txt -o btest_output_f32_n100_line_nmi -m nmi`

For mixed data (categorical, continuous data) btest automatically uses NMI as simialrity metric.
Let's generate some mixed data:

`datasim -f 32 -n 100 -a mixed -d uniform -s balanced -o btest_data_f32_n100_mixed`

Run btest od the data:

`btest -X btest_data_f32_n100_mixed/X_mixed_32_100.txt -Y btest_data_f32_n100_mixed/Y_mixed_32_100.txt -o btest_output_f32_n100_mixed`

If you try mixed data, btest provides a warning and ends as Spearman does NOT work with noncontinuous data. 

```
usage: datasim [-h] [-v] [-f FEATURES] [-n SAMPLES] [-a ASSOCIATION]
                 [-d DISTRIBUTION] [-b NOISE_BETWEEN] [-w NOISE_WITHIN] -o
                 OUTPUT [-s STRUCTURE]

btest synthetic data generator to produce paired data sets with association among their features.

optional arguments:
  -h, --help            show this help message and exit
  -v, --verbose         additional output is printed
  -f FEATURES, --features FEATURES
                        number of features in the input file D*N, Rows: D features and columns: N samples 
  -n SAMPLES, --samples SAMPLES
                        number of samples in the input file D*N, Rows: D features and columns: N samples 
  -a ASSOCIATION, --association ASSOCIATION
                        association type [sine, parabola, log, line, L, step, happy_face, default =parabola] 
  -d DISTRIBUTION, --distribution DISTRIBUTION
                        Distribution [normal, uniform, default =uniform] 
  -b NOISE_BETWEEN, --noise-between NOISE_BETWEEN
                        number of samples in the input file D*N, Rows: D features and columns: N samples 
  -w NOISE_WITHIN, --noise-within NOISE_WITHIN
                        number of samples in the input file D*N, Rows: D features and columns: N samples 
  -o OUTPUT, --output OUTPUT
                        the output directory
  -s STRUCTURE, --structure STRUCTURE
                        structure [balanced, imbalanced, default =balanced] 
```

### btest Python API ###
btest function along with command line can be called from other programs using Python API we provide an example is demonstrated here to show how to import and use `btesttest` function :

```
#!python

from btest.btest import btesttest

def main():
    btest(X='/path/to/first/datase/X.txt',\
               Y= '/path/to/second/datase/Y.txt',\
               output_dir='/path/to/btest/output/btest_output_demo')

if __name__ == "__main__":
    main( )  
```



### Nonparametric p-value ###
We have implemented both empirical cumulative distribution function (ECDF) and fast and accurate approach, generalized Pareto distribution (GPD) by Knijnenburg et al. 2009, permutation test. The function can be imported to other python programs :

```
#!python

from btest.stats import permutation_test_pvalue 
import numpy as np

def main():
    
    # Generate a list of random values for first vector
    np.random.seed(0)
    x_rand = np.random.rand(1,10)[0]   
    
    # Generate a list of random values for second vector
    # set the numpy seed for different random values from the first set
    np.random.seed(1)
    y_rand = np.random.rand(1, 10)[0]
    
    # Calculate pvalue using empirical cumulative distribution function (ECDF) 
    p_random_ecdf = permutation_test_pvalue(X  = x_rand, Y = y_rand, similarity_method = 'spearman',permutation_func = 'ecdf')
    p_perfect_ecdf = permutation_test_pvalue(X  = x_rand, Y = x_rand, similarity_method = 'spearman', permutation_func = 'ecdf')
    print ("ECDF P-value for random data: %s, ECDF P-value for perfect correlation data: %s")%(p_random_ecdf, p_perfect_ecdf)
    
    # Calculate pvalue using our implementation in btest for generalized Pareto distribution (GPD) approach proposed by Knijnenburg et al. 2009 
    p_random_gpd = permutation_test_pvalue(X  = x_rand, Y = y_rand, similarity_method = 'spearman',permutation_func = 'gpd')
    p_perfect_gpd = permutation_test_pvalue(X  = x_rand, Y = x_rand, similarity_method = 'spearman', permutation_func = 'gpd')
    print ("GPD P-value for random data: %s, GPD P-value for perfect correlation data: %s")%(p_random_gpd, p_perfect_gpd)

if __name__ == "__main__":
    main( ) 
```
The parameters that can be provided to the permutation test for calculating p-value are:

* `iterations`: the number permutation for the test (i.e. 1000)
* `permutation_func` can be either 'ECDF' or 'GPD'
* `similarity_method` a similarity metric supported by btest (check what are the choices by 'btest -h')
* `seed` if -1 each run seeds a random value, 0 uses the same seed any place does permutation.



##### Complete btest option list #####
```
usage: btest [-h] [--version] -X <input_dataset_1.txt> [-Y <input_dataset_2.txt>] -o <output> [-q <.1>] [--fnt <.25>] [-p {ecdf,gpd,none}] [-a {block,pair}] [-i <1000>] [-m {nmi,ami,mic,dmic,dcor,pearson,spearman,r2,chi,mi}]
             [--fdr {bh,by,y,meinshausen,bonferroni,no_adjusting}] [-v VERBOSE] [--diagnostics-plot] [--discretizing {equal-freq,hclust,jenks,none}] [--linkage {single,average,complete,weighted}] [--generate-one-null-samples] [--header]
             [--format-feature-names] [--nproc <1>] [--nbin <None>] [-s SEED] [-e ENTROPY_THRESHOLD] [-e1 ENTROPY_THRESHOLD1] [-e2 ENTROPY_THRESHOLD2] [--missing-char MISSING_CHAR] [--fill-missing {mean,median,most_frequent}]
             [--missing-data-category] [--write-hypothesis-tree] [-t {log,sqrt,arcsin,arcsinh,}]

btest: block-wise association testing

optional arguments:
  -h, --help            show this help message and exit
  --version             show program's version number and exit
  -X <input_dataset_1.txt>
                        first file: Tab-delimited text input file, one row per feature, one column per measurement
                        [REQUIRED]
  -Y <input_dataset_2.txt>
                        second file: Tab-delimited text input file, one row per feature, one column per measurement
                        [default = the first file (-X)]
  -o <output>, --output <output>
                        directory to write output files
                        [REQUIRED]
  -q <.1>, --q-value <.1>
                        q-value for overall significance tests (cut-off for false discovery rate)
                        [default = 0.1]
  --fnt <.25>           Estimated False Negative Tolerance in block association
                        [default = 0.25]
  -p {ecdf,gpd,none}, --permutation {ecdf,gpd,none}
                        permutation function 
                        [default = none for Spearman and Pearson and gpd for other]
  -a {block,pair}, --descending {block,pair}
                        descending approach
                        [default = block for block-wise association testing]
  -i <1000>, --iterations <1000>
                        iterations for nonparametric significance testing (permutation test)
                        [default = 1000]
  -m {nmi,ami,mic,dmic,dcor,pearson,spearman,r2,chi,mi}, --metric {nmi,ami,mic,dmic,dcor,pearson,spearman,r2,chi,mi}
                        metric to be used for similarity measurement
                        [default = '']
  --fdr {bh,by,y,meinshausen,bonferroni,no_adjusting}
                        approach for FDR correction
                        [default = bh]
  -v VERBOSE, --verbose VERBOSE
                        additional output is printed
  --diagnostics-plot    Diagnostics plot for associations 
  --discretizing {equal-freq,hclust,jenks,none}
                        approach for discretizing continuous data
                        [default = equal-freq]
  --linkage {single,average,complete,weighted}
                        The method to be used in linkage hierarchical clustering.
  --generate-one-null-samples, --fast
                        Use one null distribution for permutation test
  --header              the input files contain a header line
  --format-feature-names
                        Replaces special characters and for OTUs separated  by | uses the known end of a clade
  --nproc <1>           the number of processing units available
                        [default = 1]
  --nbin <None>         the number of bins for discretizing 
                        [default = None]
  -s SEED, --seed SEED  a seed number to make the random permutation reproducible
                        [default = 0,and -1 for random number]
  -e ENTROPY_THRESHOLD, --entropy ENTROPY_THRESHOLD
                        Minimum entropy threshold to filter features with low information
                        [default = 0.5]
  -e1 ENTROPY_THRESHOLD1, --entropy1 ENTROPY_THRESHOLD1
                        Minimum entropy threshold for the first dataset 
                        [default = None]
  -e2 ENTROPY_THRESHOLD2, --entropy2 ENTROPY_THRESHOLD2
                        Minimum entropy threshold for the second dataset 
                        [default = None]
  --missing-char MISSING_CHAR
                        defines missing characters
                        [default = '']
  --fill-missing {mean,median,most_frequent}
                        defines missing strategy to fill missing data.
                        For categorical data puts all missing data in one new category.
  --missing-data-category
                        To count the missing data as a category
  --write-hypothesis-tree
                        To write levels of hypothesis tree in the file
  -t {log,sqrt,arcsin,arcsinh,}, --transform {log,sqrt,arcsin,arcsinh,}
                        data transformation method 
                        [default = '' ]
```
