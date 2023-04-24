"""
btest setup
To run: python3 setup.py install
"""

import os
try:
    from setuptools import setup, find_packages
except ImportError:
    sys.exit("Please install setuptools.")


import urllib
try:
       from urllib.request import urlretrieve
       classifiers=[
        "Programming Language :: Python",
        "Development Status :: 5 - Production/Stable",
        "Environment :: Console",
        "Operating System :: MacOS",
        "Operating System :: Unix",
        "Operating System :: Microsoft :: Windows",
        "Programming Language :: Python :: 2.7",
        "Topic :: Scientific/Engineering :: Bio-Informatics"
        ]
except ImportError:
       from urllib import urlretrieve
       classifiers=[
        "Programming Language :: Python",
        "Development Status :: 5 - Production/Stable",
        "Environment :: Console",
        "Operating System :: MacOS",
        "Operating System :: Unix",
        "Operating System :: Microsoft :: Windows",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3.4",
        "Topic :: Scientific/Engineering :: Bio-Informatics"
        ]

VERSION = "1.1.2"
AUTHOR = "Bahar Sayoldin"
AUTHOR_EMAIL = "bahar.sayoldin@gmail.com"
MAINTAINER = "Ali Rahnavard"
MAINTAINER_EMAIL = "gholamali.rahnavard@gmail.com"

# try to download the counter file to count downloads
# this has been added since PyPI has turned off the download stats
# this will be removed when PyPI Warehouse is production as it
# will have download stats
COUNTER_URL="http://github/omicsEye/btest/downloads/README.txt"
counter_file="README.txt"
if not os.path.isfile(counter_file):
    print("Downloading counter file to track btest downloads"+
        " since the global PyPI download stats are currently turned off.")
    try:
        file, headers = urlretrieve(COUNTER_URL,counter_file)
    except EnvironmentError:
        print("Unable to download counter")

setup(
    name="btest",
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    version=VERSION,
    license="MIT",
    description="btest: Block-wise Association Testing",
    long_description="btest is a programmatic tool for performing multiple association testing " + \
        "between two or more heterogeneous datasets, each containing a mixture of discrete, binary, or continuous data." ,
    url="http://rahnavard.org/btest",
    keywords=['association','discovery','test','pattern','cluster'],
    platforms=['Linux','MacOS', "Windows"],
    classifiers=classifiers,
    #long_description=open('readme.md').read(),
    install_requires=[
        "Numpy >= 1.12.1",
        "Scipy >= 0.17.0",
        "Matplotlib >= 1.5.1",
        "Scikit-learn >= 0.14.1",
        "pandas >= 1.0.4"
    ],
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'btest = btest.btest:main',
            'blockplot = btest.blockplot:main',
            'bdata = btest.datasim:main',
            'bscatter = btest.scatter:main',
            'b_scatter = btest.b_scatter:main'
        ]},
    test_suite= 'btest.tests.btest_test.main',
    zip_safe = False
 )
