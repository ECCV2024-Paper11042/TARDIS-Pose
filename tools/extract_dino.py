from __future__ import annotations
from typing import Literal, List, Sequence, Tuple

import os
import h5py
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import sklearn
import joblib
import torch.utils.data as td
import albumentations as alb
from albumentations.pytorch import transforms as alb_torch
from mmpose.datasets.transforms import LoadImage, TopdownAffine, GetBBoxCenterScale, RandomBBoxTransform
from mmpose.datasets.datasets.animal.ap10k.ap10k_dataset import AP10KDataset

from sklearn.decomposition import PCA
import kornia


dinov2_archs = {
    "small": "vits14",
    "base": "vitb14",
    "large": "vitl14",
    "giant": "vitg14",
}


dino_archs = {
    "small": "vit_small",
    "base": "vit_base",
    "large": "vit_large",
}


dino_channels = {
    "small": 384,
    "base": 768,
    "large": 1024,
}


def get_pretrained_dino(arch, patch_size):
    import pose.models.vision_transformer as vits
    model = vits.__dict__[arch](patch_size=patch_size, num_classes=0)
    for p in model.parameters():
        p.requires_grad = False

    url = None
    if arch == "vit_small" and patch_size == 16:
        url = "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
    elif arch == "vit_small" and patch_size == 8:
        # model used for visualizations in our paper
        url = "dino_deitsmall8_300ep_pretrain/dino_deitsmall8_300ep_pretrain.pth"
    elif arch == "vit_base" and patch_size == 16:
        url = "dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
    elif arch == "vit_base" and patch_size == 8:
        url = "dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
    if url is not None:
        print("Since no pretrained weights have been provided, we load the reference pretrained DINO weights.")
        state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)
        model.load_state_dict(state_dict, strict=True)
    else:
        print("There is no reference weights available for this model => We use random weights.")
    return model


def get_pretrained_dinov2(arch) -> torch.nn.Module:
    backbone_archs = {
        "small": "vits14",
        "base": "vitb14",
        "large": "vitl14",
        "giant": "vitg14",
    }
    assert arch in backbone_archs.values()
    # backbone_arch = backbone_archs[arch]
    backbone_name = f"dinov2_{arch}"

    model = torch.hub.load(repo_or_dir="facebookresearch/dinov2", model=backbone_name)
    for p in model.parameters():
        p.requires_grad = False
    return model

class PCAVis():
    def __init__(self, segment=True, bg_thresh=0.5):
        self.pca = None 
        self.pca_fg = None
        self.segment = segment
        self.bg_thresh = bg_thresh

    def load(self, filepath):
        print(f"Loading PCA models from {filepath}")
        self.pca = joblib.load(filepath+'.pca.bg.joblib')
        if self.segment:
            self.pca_fg = joblib.load(filepath+'.pca.fg.joblib')

    def save(self, filepath):
        print(f"Saving PCA models to {filepath}")
        joblib.dump(self.pca, filepath+'.pca.bg.joblib')
        if self.pca_fg is not None:
            joblib.dump(self.pca_fg, filepath+'.pca.fg.joblib')

    @staticmethod
    def _plot_components(pca_features):
        """ visualize PCA components for finding a proper threshold 3 histograms for 3 components"""
        plt.subplot(2, 2, 1)
        plt.hist(pca_features[:, 0])
        plt.subplot(2, 2, 2)
        plt.hist(pca_features[:, 1])
        plt.subplot(2, 2, 3)
        plt.hist(pca_features[:, 2])
        plt.show()
        plt.close()

    @staticmethod
    def _show_component(pca_features, comp_id,  H, W):
        for i in range(len(pca_features)):
            plt.subplot(2, 2, i + 1)
            plt.imshow(pca_features[i * H * W: (i + 1) * H * W, comp_id].reshape(H, W))
        plt.show()
        plt.close()

    def fit(self, features: np.ndarray | torch.Tensor):
        features = to_numpy(features)

        N, D, H, W = features.shape
        features = features.transpose(0, 2, 3, 1).reshape(-1, D)

        # print(f"Fitting PCA for background segmentation...")
        self.pca = PCA(n_components=3)
        self.pca.fit(features)

        if self.segment:
            pca_features = self.pca.transform(features)
            pca_features = self._minmax_scale(pca_features)

            # self._plot_components(pca_features)
            # self._show_component(pca_features[:4], 0, H, W)

            # segment/seperate the backgound and foreground using the first component
            pca_features_bg = pca_features[:, 0] < self.bg_thresh  # from first histogram
            pca_features_fg = ~pca_features_bg

            if False:
                # plot the pca_features_bg
                for i in range(4):
                    plt.subplot(2, 2, i + 1)
                    plt.imshow(pca_features_bg[i * H * W: (i + 1) * H * W].reshape(H, W))
                plt.show()
                plt.close()

            # print(f"Fitting PCA for only foreground patches...")
            self.pca_fg = PCA(n_components=3)
            self.pca_fg.fit(features[pca_features_fg])

    @staticmethod
    def _minmax_scale(features):
        features = sklearn.preprocessing.minmax_scale(features)
        return features

    def transform(self, features: np.ndarray | torch.Tensor):
        features = to_numpy(features)

        input_ndim = len(features.shape)
        if input_ndim == 3:
            features = features[np.newaxis]

        if self.pca is None:
            self.fit(features)

        N, D, H, W = features.shape

        if D == 9:  # or D == 32:
            features = features[:, :3]
            D = 3

        if D == 3:
            pca_features = features.transpose(0, 2, 3, 1)

            f = pca_features
            for i in range(3):
                f[..., i] = (f[...,i] - f[...,i].min()) / (f[..., i].max() - f[...,i].min())
            return (255 * f).astype(np.uint8)

        features = features.transpose(0, 2, 3, 1).reshape(-1, D)

        pca_features = self.pca.transform(features)
        pca_features = self._minmax_scale(pca_features)

        if self.segment:
            # segment/seperate the backgound and foreground using the first component
            pca_features_bg = pca_features[:, 0] < self.bg_thresh
            pca_features_fg = ~pca_features_bg

            pca_features_left = self.pca_fg.transform(features[pca_features_fg])
            pca_features_left = self._minmax_scale(pca_features_left)

            pca_features_rgb = pca_features.copy()
            pca_features_rgb[pca_features_bg] = 0
            pca_features_rgb[pca_features_fg] = pca_features_left
        else:
            pca_features_rgb = pca_features

        # reshaping to numpy image format
        pca_features_rgb = pca_features_rgb.reshape(N, H, W, 3)

        if False:
            nimgs = min(10, N)
            for i in range(nimgs):
                plt.subplot(min(2, nimgs), nimgs // min(2, nimgs), i + 1)
                plt.imshow(pca_features_rgb[i])
            plt.show()
            plt.close()

        if input_ndim == 3:
            pca_features_rgb = pca_features_rgb[0]

        return (pca_features_rgb * 255).astype(np.uint8)


def get_features_from_hdf5(filepath, num):
    print(f"Loading DINO features from file {filepath}")

    features = []
    with h5py.File(filepath, 'r') as f:
        ids = np.random.choice(list(f.keys()), num, replace=False)
        # D, H, W = np.array(f[str(id)]).shape
        # features = np.zeros(len(ids), D, H, W)
        for id in ids:
            features.append(np.array(f[str(id)]))

    features = np.array(features)
    print(f"Features loaded: {features.shape}")
    return features


def _extract_dino_features(
        model: torch.nn.Module, images: torch.Tensor, patch_size: int, dino_version: str = 'dinov2'
) -> torch.Tensor:
    N, C, H, W = images.shape
    assert H % patch_size == 0
    assert W % patch_size == 0

    w_featmap = W // patch_size
    h_featmap = H // patch_size

    if dino_version == 'dino':
        features = model.get_last_selfattention(images)[:, :, 0, 1:]
    elif dino_version == 'dinov2':
        with torch.no_grad():
            features = model.forward_features(images)['x_norm_patchtokens']
        features = features.permute(0, 2, 1)
    else:
        raise ValueError(f"Unknown dino version {dino_version}!")

    assert features.ndim == 3
    nh = features.shape[1]
    return features.reshape(N, nh, w_featmap, h_featmap)


def extract(dataloader, filepath, dino_version, arch, target_size, patch_size, extract_size):

    if dino_version == 'dinov2':
        dino = get_pretrained_dinov2(arch=arch).eval().cuda()
    else:
        dino = get_pretrained_dino(arch=arch, patch_size=patch_size).eval().cuda()

    def fliplr(tensor: torch.Tensor) -> torch.Tensor:
        N, C, H, W = tensor.shape
        return torch.fliplr(tensor.reshape(-1, W)).reshape(N, C, H, W)

    def resize(tensor: torch.Tensor, size: Tuple) -> torch.Tensor:
        images_resized = torch.nn.functional.interpolate(tensor, size=size, mode='bilinear')
        return images_resized

    def average(tensors):
        return torch.mean(torch.stack(tensors), dim=0)

    def extract(images, extract_size, output_size):
        _new_img_size = (extract_size[0] * patch_size, extract_size[1] * patch_size)
        _resized_images = resize(images, size=_new_img_size).cuda()
        features = _extract_dino_features(dino, _resized_images, patch_size, dino_version)
        return resize(features, size=output_size).detach().cpu()

    def extract_with_flip(images, extract_size, output_size):
        f1 = extract(images, extract_size, output_size)
        f2 = fliplr(extract(fliplr(images), extract_size, output_size))
        return average([f1, f2]).detach().cpu()

    if os.path.isfile(filepath):
        raise IOError(f"Output file {filepath} already exists.")

    os.makedirs('data/dino', exist_ok=True)

    with h5py.File(filepath, 'w') as h5f:
        for iter_, data in enumerate(dataloader):
            images = data['image']

            a = extract_with_flip(images, extract_size=extract_size, output_size=target_size).cpu().numpy()

            if False:
                def show_pca(a):
                    pcavis = PCAVis(segment=False)
                    pcavis.fit(a)
                    pca_features_rgb = pcavis.transform(a)
                    fig = plt.figure(figsize=(20, 16))
                    nimgs = min(4, a.shape[0])
                    for i in range(nimgs):
                        ax = fig.add_subplot(2, nimgs, i + 1)
                        ax.imshow(vis.to_disp_image(images[i]))
                    for i in range(nimgs):
                        ax = fig.add_subplot(2, nimgs, i + 1 + nimgs)
                        ax.imshow(pca_features_rgb[i])

                show_pca(a)
                plt.tight_layout()
                plt.show()
                plt.close()

            assert(a.shape[2:] == target_size)
            for i, idx in enumerate(data['id']):
                idx_key = str(int(idx))
                try:
                    h5f.create_dataset(idx_key, data=a[i])
                except ValueError:
                    print(f"Duplicate index {idx_key}!")

            if (iter_+1) % 1 == 0:
                print(f"[{iter_+1}/{len(dataloader)}]")


if __name__ == '__main__':

    # dino_version = 'dino'
    # arch = dino_archs['base']
    # patch_size = 8
    # dino_channels = 6
    # image_size = (112 * patch_size, 112 * patch_size)

    dino_version = 'dinov2'
    patch_size = 14
    model_size = 'large'
    arch = dinov2_archs[model_size]
    dino_channels = dino_channels[model_size]
    image_size = (64 * patch_size, 64 * patch_size)

    dataset_name = 'ap10k'
    split = 'train-split1'

    ap10k_root = os.path.join(os.getcwd(), 'datasets/animal_data/ap-10k')
    ap10k_meta_file = './mmpose/datasets/datasets/animal/ap10k/ap10k.py'

    dataset = AP10KDataset(
        ann_file=os.path.join(ap10k_root, f"annotations/ap10k-{split}.json"),
        data_root=os.path.join(ap10k_root, "data"),
        metainfo=dict(from_file=ap10k_meta_file),
        pipeline=[
            LoadImage(),
            GetBBoxCenterScale(padding=1.25),
            TopdownAffine(input_size=image_size),
        ],
        transform=alb.Compose([alb.Normalize(), alb_torch.ToTensorV2()]),
        test_mode=True
    )
    dl = td.DataLoader(dataset, batch_size=2, num_workers=4, shuffle=False)

    out_dir = './data/dino'
    outfile = os.path.join(out_dir, f'dino_{dataset_name}_{split}_{arch}_{patch_size}.h5')
    extract(dl, outfile, dino_version, arch, target_size=(64, 64), patch_size=patch_size, extract_size=(64, 64))