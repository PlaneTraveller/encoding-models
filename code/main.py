#!/usr/bin/env python3

import os
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# <why>
import matplotlib
from matplotlib import pyplot as plt

from nilearn import datasets
from nilearn import plotting

import torch
from torch.utils.data import Dataset, DataLoader

# <why>
from torchvision.models.feature_extraction import (
    create_feature_extractor,
    get_graph_node_names,
)
from torchvision import transforms

from sklearn.decomposition import IncrementalPCA
from sklearn.linear_model import LogisticRegression

from scipy.stats import pearsonr as corr

# Custom scripts
from logger import logger


# ===============================================================================
# = Variables & Options


data_dir = "../data"
submission_dir = "../submission"

device = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device)

log = logger("../logs")


# ===============================================================================
# = Data Wrangling


# argument parser for subjects
class argObj:
    def __init__(self, data_dir, submission_dir, subj):
        self.subj = format(subj, "02")
        self.data_dir = os.path.join(data_dir, "subj" + self.subj)
        self.submission_dir = submission_dir
        self.subj_submission_dir = os.path.join(submission_dir, "subj" + self.subj)

        # create directory if not created
        if not os.path.isdir(self.subj_submission_dir):
            os.makedirs(self.subj_submission_dir)


def wrangle_data():
    # the subject to wrangle with
    subj = 1
    arg = argObj(data_dir, submission_dir, subj)
    fmri_dir = os.path.join(arg.data_dir, "training_split", "training_fmri")
    lh_fmri = np.load(os.path.join(fmri_dir, "lh_training_fmri.npy"))
    rh_fmri = np.load(os.path.join(fmri_dir, "rh_training_fmri.npy"))

    img_dir = os.path.join(arg.data_dir, "training_split", "training_images")

    count_files(img_dir)

    print(lh_fmri.shape)

    pass


# ===============================================================================
# = Helper Functions


def count_files(dir):
    count = 0
    for path in os.listdir(dir):
        if os.path.isfile(os.path.join(dir, path)):
            count += 1
    print(count)
    return count


# ===============================================================================
# = Main


def main():
    wrangle_data()

    pass


# ===============================================================================
# = Run


if __name__ == "__main__":
    main()
