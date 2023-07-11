#!/usr/bin/env python3

import os
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# <why>
import matplotlib
from matplotlib import pyplot as plt

from nilearn import datasets, plotting

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
        self.subj = format(subj, "02d")
        self.data_dir = os.path.join(data_dir, "subj" + self.subj)
        self.submission_dir = submission_dir
        self.roi_dir_base = os.path.join(self.data_dir, "roi_masks")

        self.subj_submission_dir = os.path.join(submission_dir, "subj" + self.subj)

        # == create directory if not created
        if not os.path.isdir(self.subj_submission_dir):
            os.makedirs(self.subj_submission_dir)

    def get_roi_dir(self, hemisphere, file_name):
        if hemisphere == "":
            return os.path.join(self.roi_dir_base, file_name + ".npy")
        return os.path.join(
            self.roi_dir_base, hemisphere[0] + "h." + file_name + ".npy"
        )


def wrangle_data():
    # == navigation
    subj = 1
    arg = argObj(data_dir, submission_dir, subj)
    fmri_dir = os.path.join(arg.data_dir, "training_split", "training_fmri")
    lh_fmri = np.load(os.path.join(fmri_dir, "lh_training_fmri.npy"))
    rh_fmri = np.load(os.path.join(fmri_dir, "rh_training_fmri.npy"))

    img_dir = os.path.join(arg.data_dir, "training_split", "training_images")

    # == observing the data
    # count_files(img_dir)
    # print(lh_fmri.shape)
    # print(rh_fmri)

    # == ROI indexing
    hemisphere = "left"
    roi_dir = arg.get_roi_dir(hemisphere[0] + "h", "all-vertices_fsaverage_space")
    fsaverage_all_vertices = np.load(roi_dir)

    # == visualizing brain surface map
    fsaverage = datasets.fetch_surf_fsaverage("fsaverage")
    view = plotting.view_surf(
        surf_mesh=fsaverage["infl_" + hemisphere],
        surf_map=fsaverage_all_vertices,
        bg_map=fsaverage["sulc_" + hemisphere],
        threshold=1e-14,
        cmap="cool",
        colorbar=False,
        title="All vertices, " + hemisphere[0] + "h" + " hemisphere",
    )

    view.open_in_browser()

    # == visualizing ROI
    roi = "EBA"
    if roi in ["V1v", "V1d", "V2v", "V2d", "V3v", "V3d", "hV4"]:
        roi_class = "prf-visualrois"
    elif roi in ["EBA", "FBA-1", "FBA-2", "mTL-bodies"]:
        roi_class = "floc-bodies"
    elif roi in ["OFA", "FFA-1", "FFA-2", "mTL-faces", "aTL-faces"]:
        roi_class = "floc-faces"
    elif roi in ["OPA", "PPA", "RSC"]:
        roi_class = "floc-places"
    elif roi in ["OWFA", "VWFA-1", "VWFA-2", "mfs-words", "mTL-words"]:
        roi_class = "floc-words"
    elif roi in [
        "early",
        "midventral",
        "midlateral",
        "midparietal",
        "ventral",
        "lateral",
        "parietal",
    ]:
        roi_class = "streams"
    else:
        roi = ""
        raise ValueError("ROI not found")

    roi_class_dir = arg.get_roi_dir(hemisphere, roi_class)
    roi_mapping_dir = arg.get_roi_dir("", "roi_mapping")

    fsaverage_roi_class = np.load(roi_class_dir)
    roi_map = np.load(roi_mapping_dir, allow_pickle=True).item()

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


def visualize_roi(roi, hemisphere, title):
    """
    ROI Options:
    "V1v", "V1d", "V2v", "V2d", "V3v", "V3d", "hV4", "EBA", "FBA-1", "FBA-2", "mTL-bodies", "OFA", "FFA-1", "FFA-2", "mTL-faces", "aTL-faces", "OPA", "PPA", "RSC", "OWFA", "VWFA-1", "VWFA-2", "mfs-words", "mTL-words", "early", "midventral", "midlateral", "midparietal", "ventral", "lateral", "parietal"
    """
    if roi in ["V1v", "V1d", "V2v", "V2d", "V3v", "V3d", "hV4"]:
        roi_class = "prf-visualrois"
    elif roi in ["EBA", "FBA-1", "FBA-2", "mTL-bodies"]:
        roi_class = "floc-bodies"
    elif roi in ["OFA", "FFA-1", "FFA-2", "mTL-faces", "aTL-faces"]:
        roi_class = "floc-faces"
    elif roi in ["OPA", "PPA", "RSC"]:
        roi_class = "floc-places"
    elif roi in ["OWFA", "VWFA-1", "VWFA-2", "mfs-words", "mTL-words"]:
        roi_class = "floc-words"
    elif roi in [
        "early",
        "midventral",
        "midlateral",
        "midparietal",
        "ventral",
        "lateral",
        "parietal",
    ]:
        roi_class = "streams"


# ===============================================================================
# = Main


def main():
    wrangle_data()

    pass


# ===============================================================================
# = Run


if __name__ == "__main__":
    main()
