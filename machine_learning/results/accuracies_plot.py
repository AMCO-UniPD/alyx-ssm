"""
Python script to produce the plot of test sequence length vs accuracy
"""

import ipdb
import os
import sys
import pandas as pd
import numpy as np
import signal
import dotenv
import hydra
from omegaconf import DictConfig

script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
ml_path = os.path.dirname(script_dir)

sys.path.append(ml_path)

from src.utils import utils

# load environment variables from `.env` file if it exists
# recursively searches for `.env` in all folders starting from work dir
dotenv.load_dotenv(override=True)

log = utils.get_logger(__name__)

@hydra.main(config_path="configs/", config_name="results_config.yaml", version_base="1.1")
def main(config: DictConfig):

    from src.utils import utils

    results_dirpath = utils.get_results_path(config=config)
    mean_acc_df_path = os.path.join(results_dirpath,"mean_acc_df.csv")
    mean_acc_df = pd.read_csv(mean_acc_df_path)

    #TODO: To execute this script I need to save the run results
    # with a different directory naming → I need to have model_name/data_encoding/seed

    #TODO: Use the same results_config.yaml of get_mean_accuracies
    # - add metric value to see weather plotting top1,top2,top3,...
    # - use the quantile stats to do plt.fill_between
