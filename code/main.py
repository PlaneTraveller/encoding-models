#!/usr/bin/env python3

# ===============================================================================
# = Imports


import os
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm

import matplotlib
from matplotlib import pyplot as plt

from nilearn import datasets, plotting

import torch
from torch.utils.data import Dataset, DataLoader

from torchvision.models.feature_extraction import (
    create_feature_extractor,
    get_graph_node_names,
)
from torchvision import transforms

from sklearn.decomposition import IncrementalPCA
from sklearn.linear_model import LinearRegression

from scipy.stats import pearsonr as corr

# == Custom scripts and model
from logger import logger


# ===============================================================================
# = Variables & Options


data_dir = "../data"
submission_dir = "../submission"

device = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device)

log = logger("../logs")


# ===============================================================================
# = Classes


class Subject:
    """Class for each subject."""

    def __init__(self, data_dir, submission_dir, subj):
        self.subj = format(subj, "02d")
        self.data_dir = os.path.join(data_dir, "subj" + self.subj)
        self.submission_dir = submission_dir
        self.roi_dir_base = os.path.join(self.data_dir, "roi_masks")

        self.subj_submission_dir = os.path.join(submission_dir, "subj" + self.subj)

        self.fmri_dir = os.path.join(self.data_dir, "training_split", "training_fmri")
        self.lh_fmri = np.load(os.path.join(self.fmri_dir, "lh_training_fmri.npy"))
        self.rh_fmri = np.load(os.path.join(self.fmri_dir, "rh_training_fmri.npy"))

        self.img_dir = os.path.join(self.data_dir, "training_split", "training_images")
        self.test_img_dir = os.path.join(self.data_dir, "test_split", "test_images")

        self.imgs_paths = sorted(list(Path(self.img_dir).iterdir()))
        self.test_imgs_paths = sorted(list(Path(self.test_img_dir).iterdir()))

        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),  # resize the images to 224x24 pixels
                transforms.ToTensor(),  # convert the images to a PyTorch tensor
                transforms.Normalize(
                    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                ),  # normalize the images color channels
            ]
        )

        # == create directory if not created
        if not os.path.isdir(self.subj_submission_dir):
            os.makedirs(self.subj_submission_dir)

    def get_roi_dir(self, location, file_name, challenge=True):
        """Gets ROI mask file directory based on arguments."""
        ret = self.roi_dir_base
        if location == "" or location == "mapping_":
            ret = os.path.join(ret, location + file_name)
        elif location == "left" or location == "right":
            ret = os.path.join(ret, location[0] + "h." + file_name)
            if not challenge:
                ret = ret + "_fsaverage_space"
            else:
                ret = ret + "_challenge_space"
        else:
            raise ValueError("Location not found")

        ret = ret + ".npy"
        return ret

    def get_roi_mask(self, roi, hemisphere, vis=False, challenge=True):
        """Gets ROI mask based on arguments."""
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
        elif roi == "all":
            roi_class = "all-vertices"
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
            raise ValueError("ROI not found")

        # == selecting vertices
        # roi_class corresponds to file_name

        roi_class_dir = self.get_roi_dir(hemisphere, roi_class, challenge)
        if roi == "all":
            fsaverage_roi = np.load(roi_class_dir)

        else:
            roi_mapping_dir = self.get_roi_dir("", "mapping_" + roi_class, challenge)

            fsaverage_roi_class = np.load(roi_class_dir)
            roi_map = np.load(roi_mapping_dir, allow_pickle=True).item()

            # find index of roi
            # fsaverage_roi_class is an array of voxels, each being one of 0, 1, 2, 3, indicating the roi
            roi_mapping = list(roi_map.keys())[list(roi_map.values()).index(roi)]

            # fsaverage_roi is an array of booleans, indicating which voxels are in the roi
            fsaverage_roi = np.asarray(fsaverage_roi_class == roi_mapping, dtype=int)

        if vis:
            fsaverage = datasets.fetch_surf_fsaverage("fsaverage")
            view = plotting.view_surf(
                surf_mesh=fsaverage["infl_" + hemisphere],
                surf_map=fsaverage_roi,
                bg_map=fsaverage["sulc_" + hemisphere],
                threshold=1e-14,
                cmap="cool",
                colorbar=False,
                title=roi + ", " + hemisphere + " hemisphere",
            )
            view.open_in_browser()

        return fsaverage_roi

    def get_img_dir_by_idx(self, idx):
        """Gets image by index."""
        img_list = os.listdir(self.img_dir)
        img_list.sort()
        img_file = img_list[idx]
        # img = np.load(os.path.join(self.img_dir, img_file))
        ret = os.path.join(self.img_dir, img_file)
        return ret

    def get_split_idxs(self, val_split, seed=666):
        rand_seed = seed
        np.random.seed(rand_seed)

        img_list = os.listdir(self.img_dir)
        test_img_list = os.listdir(self.test_img_dir)
        num_train = int(np.round(len(img_list) * (1 - val_split)))

        # Obtaining training index and validation index
        # train_idx = np.random.choice(len(img_list), num_train, replace=False)
        # val_idx = np.setdiff1d(np.arange(len(img_list)), train_idx)

        idxs = np.arange(len(img_list))
        np.random.shuffle(idxs)
        train_idx, val_idx = idxs[:num_train], idxs[num_train:]
        test_idx = np.arange(len(test_img_list))

        ret = {"train": train_idx, "val": val_idx, "test": test_idx}
        return ret

    def get_split_data(self, val_split, batch_size=500, seed=666):
        data_idxs_dict = self.get_split_idxs(val_split, seed)
        train_idx = data_idxs_dict["train"]
        val_idx = data_idxs_dict["val"]
        test_idx = data_idxs_dict["test"]

        train_imgs_dataloader = DataLoader(
            ImageDataset(self.imgs_paths, train_idx, self.transform),
            batch_size=batch_size,
        )
        val_imgs_dataloader = DataLoader(
            ImageDataset(self.imgs_paths, val_idx, self.transform),
            batch_size=batch_size,
        )
        test_imgs_dataloader = DataLoader(
            ImageDataset(self.test_imgs_paths, test_idx, self.transform),
            batch_size=batch_size,
        )

        # fMRI data
        lh_fmri_train = self.lh_fmri[train_idx]
        lh_fmri_val = self.lh_fmri[val_idx]
        rh_fmri_train = self.rh_fmri[train_idx]
        rh_fmri_val = self.rh_fmri[val_idx]

        ret = {
            "train": train_imgs_dataloader,
            "val": val_imgs_dataloader,
            "test": test_imgs_dataloader,
            "lh_train": lh_fmri_train,
            "lh_val": lh_fmri_val,
            "rh_train": rh_fmri_train,
            "rh_val": rh_fmri_train,
        }
        return ret


class ImageDataset(Dataset):
    def __init__(self, imgs_paths, idxs, transform):
        self.imgs_paths = np.array(imgs_paths)[idxs]
        self.transform = transform

    def __len__(self):
        return len(self.imgs_paths)

    def __getitem__(self, idx):
        # Load the image
        img_path = self.imgs_paths[idx]
        img = Image.open(img_path).convert("RGB")
        # Preprocess the image and send it to the chosen device ('cpu' or 'cuda')
        if self.transform:
            img = self.transform(img).to(device)
        return img


# ===============================================================================
# = Data Wrangling


def wrangle_data():
    # == navigation
    subj = 1
    subject = Subject(data_dir, submission_dir, subj)
    hemisphere = "left"

    # == data shape
    # count_files(subject.img_dir)
    # print(subject.lh_fmri.shape)
    # print(subject.rh_fmri)

    # == visualizing all ROI
    # all_roi = subject.get_roi_mask("all", hemisphere, True, False)
    # print(all_roi)

    # == visualizing specific ROI
    # roi_mask = subject.get_roi_mask("EBA", hemisphere, True, False)
    # print(roi_mask)

    # == image shape
    # img_list = os.listdir(subject.img_dir)
    # img_list.sort()
    # print(str(len(img_list)))
    # img_file = img_list[0]
    # print(img_file)
    # print("NSD imgID: ", img_file[-9:-4])

    # == visualizing GT fMRI responses
    # img_idx = 0
    # img = Image.open(subject.get_img_dir_by_idx(img_idx)).convert("RGB")

    # roi_dir = subject.get_roi_dir(hemisphere, "all-vertices", False)
    # fsaverage_all_vertices = np.load(roi_dir)

    # lh_fmri0 = subject.lh_fmri[img_idx]
    # rh_fmri0 = subject.rh_fmri[img_idx]

    # fsaverage_response = np.zeros(len(fsaverage_all_vertices))
    # if hemisphere == "left":
    #     fsaverage_response[fsaverage_all_vertices == 1] = lh_fmri0
    # elif hemisphere == "right":
    #     fsaverage_response[fsaverage_all_vertices == 1] = rh_fmri0

    # fsaverage = datasets.fetch_surf_fsaverage("fsaverage")
    # view = plotting.view_surf(
    #     surf_mesh=fsaverage["infl_" + hemisphere],
    #     surf_map=fsaverage_response,
    #     bg_map=fsaverage["sulc_" + hemisphere],
    #     threshold=1e-14,
    #     cmap="cold_hot",
    #     colorbar=True,
    #     title="All vertices, " + hemisphere + " hemisphere",
    # )
    # view.open_in_browser()

    # == visualizing GT fMRI responses for specific ROI
    # roi = "EBA"
    # roi_mask = subject.get_roi_mask(roi, hemisphere, False, True)
    # roi_response = np.zeros(len(roi_mask))
    # roi_response[roi_mask == 1] = lh_fmri0[roi_mask == 1]

    # fsaverage_response = np.zeros(len(fsaverage_all_vertices))
    # fsaverage_response[fsaverage_all_vertices == 1] = roi_response

    # fsaverage = datasets.fetch_surf_fsaverage("fsaverage")
    # view = plotting.view_surf(
    #     surf_mesh=fsaverage["infl_" + hemisphere],
    #     surf_map=fsaverage_response,
    #     bg_map=fsaverage["sulc_" + hemisphere],
    #     threshold=1e-14,
    #     cmap="cold_hot",
    #     colorbar=True,
    #     title=roi + ", " + hemisphere + " hemisphere",
    # )
    # view.open_in_browser()

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


def fit_pca(feature_extractor, dataloader, batch_size):
    # Define PCA parameters
    pca = IncrementalPCA(n_components=100, batch_size=batch_size)

    # Fit PCA to batch
    for _, d in tqdm(enumerate(dataloader), total=len(dataloader)):
        # Extract features
        ft = feature_extractor(d)
        # Flatten the features
        ft = torch.hstack([torch.flatten(l, start_dim=1) for l in ft.values()])
        # Fit PCA to batch
        pca.partial_fit(ft.detach().cpu().numpy())
    return pca


def extract_features(feature_extractor, dataloader, pca):
    features = []
    for _, d in tqdm(enumerate(dataloader), total=len(dataloader)):
        # Extract features
        ft = feature_extractor(d)
        # Flatten the features
        ft = torch.hstack([torch.flatten(l, start_dim=1) for l in ft.values()])
        # Apply PCA transform
        ft = pca.transform(ft.cpu().detach().numpy())
        features.append(ft)
    return np.vstack(features)


# ===============================================================================
# = Main


def main():
    # wrangle_data()

    # == Splitting data
    rand_seed = 666
    subj = 1
    val_split = 0.1
    subject = Subject(data_dir, submission_dir, subj)

    # == Preparing dataloader
    data = subject.get_split_data(val_split)
    # train_imgs_dataloader = data["train"]
    # val_imgs_dataloader = data["val"]
    # test_imgs_dataloader = data["test"]

    # lh_fmri_train = data["lh_train"]
    # lh_fmri_val = data["lh_val"]
    # rh_fmri_train = data["rh_train"]
    # rh_fmri_val = data["rh_val"]

    # del data

    # == Preparing model
    torch.hub._validate_not_a_forked_repo = lambda a, b, c: True
    model = torch.hub.load("pytorch/vision:v0.10.0", "alexnet")
    # model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    model.to(device)
    model.eval()
    train_nodes, _ = get_graph_node_names(model)
    print(train_nodes)

    # == Preparing feature extractor
    model_layer = "features.2"
    feature_extractor = create_feature_extractor(model, return_nodes=[model_layer])

    pca_model = fit_pca(feature_extractor, data["train"], 500)

    # == Extracting features
    train_ft = extract_features(feature_extractor, data["train"], pca_model)
    val_ft = extract_features(feature_extractor, data["val"], pca_model)
    test_ft = extract_features(feature_extractor, data["test"], pca_model)

    del model, pca_model

    print("\nTraining images features:")
    print(train_ft.shape)
    print("(Training stimulus images × PCA features)")

    print("\nValidation images features:")
    print(val_ft.shape)
    print("(Validation stimulus images × PCA features)")

    print("\nTest images features:")
    print(test_ft.shape)
    print("(Test stimulus images × PCA features)")

    # <why> permutation?
    # == Linear regression mapping
    lh_reg = LinearRegression().fit(train_ft, data["lh_train"])
    rh_reg = LinearRegression().fit(train_ft, data["rh_train"])

    lh_val_pred = lh_reg.predict(val_ft)
    # rh_val_pred = rh_reg.predict(val_ft)

    # == Calculating correlation
    lh_corr = np.zeros(lh_val_pred.shape[1])
    for v in tqdm(range(lh_val_pred.shape[1])):
        lh_corr[v] = corr(lh_val_pred[:, v], data["lh_val"][:, v])[0]

    # == Visualizing encoding accuracy
    hemisphere = "left"
    fsaverage_all_vertices = np.load(
        subject.get_roi_dir(hemisphere, "all-vertices", False)
    )
    fsaverage_corr = np.zeros(len(fsaverage_all_vertices))
    # <why>
    fsaverage_corr[np.where(fsaverage_all_vertices)[0]] = lh_corr

    fsaverage = datasets.fetch_surf_fsaverage("fsaverage")
    view = plotting.view_surf(
        surf_mesh=fsaverage["infl_" + hemisphere],
        surf_map=fsaverage_corr,
        bg_map=fsaverage["sulc_" + hemisphere],
        threshold=1e-14,
        cmap="cold_hot",
        colorbar=True,
        title="Encoding accuracy, " + hemisphere + " hemisphere",
    )
    view.open_in_browser()


# ===============================================================================
# = Run


if __name__ == "__main__":
    main()
