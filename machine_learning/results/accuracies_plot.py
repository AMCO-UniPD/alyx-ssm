"""
Python script to produce the plot of test sequence length vs accuracy
"""

import ipdb
import os
import sys

script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
ml_path = os.path.dirname(script_dir)

sys.path.append(ml_path)

#TODO: Use the same results_config.yaml of get_mean_accuracies
# - add metric value to see weather plotting top1,top2,top3,...
# - use the quantile stats to do plt.fill_between
