version = '1.1.1'
__description__ = """
btest for block-wise association testing 
configuration file
"""

__doc__ = __doc__
__version__ = version
__author__ = ["Ali Rahnavard"]
__contact__ = "gholamali.rahnavard@gmail.com"

keys_attribute = ["__description__", "__version__", "__author__", "__contact__", "q", "distance", "iterations",
                  "decomposition", "p_adjust_method", "randomization_method"]

verbose = 'CRITICAL'  # "DEBUG","INFO","WARNING","ERROR","CRITICAL"
similarity_method = "spearman"
output_dir = "./"
missing_char = ''
min_var = 0.0