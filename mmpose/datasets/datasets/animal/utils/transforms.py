from __future__ import absolute_import

import os
import numpy as np
import scipy.misc
import matplotlib.pyplot as plt
import torch

from .misc import *
from .imutils import *


def color_normalize(x, mean, std):
    if x.size(0) == 1:
        x = x.repeat(3, 1, 1)

    for t, m, s in zip(x, mean, std):
        t.sub_(m)
    return x


def flip_back(flip_output, dataset):
    """
    flip output map
    """
    if dataset ==  'mpii':
        matchedParts = (
            [0,5],   [1,4],   [2,3],
            [10,15], [11,14], [12,13]
        )
    elif dataset == '_300w':
        matchedParts = ([0, 16], [1, 15], [2, 14], [3, 13], [4, 12], [5, 11], [6, 10], [7, 9],
                        [17, 26], [18, 25], [19, 26], [20, 23], [21, 22], [36, 45], [37, 44],
                        [38, 43], [39, 42], [41, 46], [40, 47], [31, 35], [32, 34], [50, 52],
                        [49, 53], [48, 54], [61, 63], [62, 64], [67, 65], [59, 55], [58, 56])
    elif dataset == 'scut':
        matchedParts = ([1, 21], [2, 20], [3, 19], [4, 18], [5, 17], [6, 16], [7, 15],
                        [8, 14], [9, 13], [10, 12], [26, 32], [25, 33], [24, 34], [23, 35],
                        [22, 36], [27, 41], [28, 40], [29, 39], [30, 38], [31, 37],
                        [49, 55], [48, 56], [47, 57], [46, 50], [45, 51], [44, 52], [43, 53], [42, 54], [58, 59],
                        [60, 72], [61, 71], [62, 70], [63, 69], [64, 68], [65, 67],
                        [79, 73], [78, 74], [77, 75], [80, 85], [81, 84], [82, 83])
    elif dataset == 'real_animal':
        matchedParts = ([0, 1], [3, 4], [5, 6], [8, 9], [10, 11], [12, 13], [14, 15], [16, 17])
    else:
        print('Not supported dataset: ' + dataset)

    # flip output horizontally
    flip_output = fliplr(flip_output.numpy())

    # Change left-right parts
    for pair in matchedParts:
        tmp = np.copy(flip_output[:, pair[0], :, :])
        flip_output[:, pair[0], :, :] = flip_output[:, pair[1], :, :]
        flip_output[:, pair[1], :, :] = tmp

    return torch.from_numpy(flip_output).float()


def shufflelr_ori(x, width, dataset):
    """
    flip coords
    """
    if dataset == 'mpii':
        matchedParts = (
            [0,5],   [1,4],   [2,3],
            [10,15], [11,14], [12,13]
        )

    elif dataset == '_300w':
        matchedParts = ([0, 16], [1, 15], [2, 14], [3, 13], [4, 12], [5, 11], [6, 10], [7, 9],
                        [17, 26], [18, 25], [19, 26], [20, 23], [21, 22], [36, 45], [37, 44],
                        [38, 43], [39, 42], [41, 46], [40, 47], [31, 35], [32, 34], [50, 52],
                        [49, 53], [48, 54], [61, 63], [62, 64], [67, 65], [59, 55], [58, 56])
    elif dataset == 'scut':
        matchedParts = ([1, 21], [2, 20], [3, 19], [4, 18], [5, 17], [6, 16], [7, 15],
                        [8, 14], [9, 13], [10, 12], [26, 32], [25, 33], [24, 34], [23, 35],
                        [22, 36], [27, 41], [28, 40], [29, 39], [30, 38], [31, 37],
                        [49, 55], [48, 56], [47, 57], [46, 50], [45, 51], [44, 52], [43, 53], [42, 54], [58, 59],
                        [60, 72], [61, 71], [62, 70], [63, 69], [64, 68], [65, 67],
                        [79, 73], [78, 74], [77, 75], [80, 85], [81, 84], [82, 83])
    elif dataset == 'real_animal':
        matchedParts = ([0, 1], [3, 4], [5, 6], [8, 9], [10, 11], [12, 13], [14, 15], [16, 17])
    else:
        print('Not supported dataset: ' + dataset)

    # Flip horizontal
    x[:, 0] = width - x[:, 0]

    # Change left-right parts
    for pair in matchedParts:
        tmp = x[pair[0], :].clone()
        x[pair[0], :] = x[pair[1], :]
        x[pair[1], :] = tmp

    return x


def fliplr(x):
    if x.ndim == 3:
        x = np.transpose(np.fliplr(np.transpose(x, (0, 2, 1))), (0, 2, 1))
    elif x.ndim == 4:
        for i in range(x.shape[0]):
            x[i] = np.transpose(np.fliplr(np.transpose(x[i], (0, 2, 1))), (0, 2, 1))
    return x.astype(float)


def flip_weights(w):
    flipparts = [1, 0, 2, 4, 3, 6, 5, 7, 9, 8, 11, 10, 13, 12, 15, 14, 17, 16]
    return w[flipparts]


def get_transform(center, scale, res, rot=0):
    """
    General image processing functions
    """
    # Generate transformation matrix
    h = 200 * scale
    t = np.zeros((3, 3))
    t[0, 0] = float(res[1]) / h
    t[1, 1] = float(res[0]) / h
    t[0, 2] = res[1] * (-float(center[0]) / h + .5)
    t[1, 2] = res[0] * (-float(center[1]) / h + .5)
    t[2, 2] = 1
    if not rot == 0:
        rot = -rot # To match direction of rotation from cropping
        rot_mat = np.zeros((3,3))
        rot_rad = rot * np.pi / 180
        sn,cs = np.sin(rot_rad), np.cos(rot_rad)
        rot_mat[0,:2] = [cs, -sn]
        rot_mat[1,:2] = [sn, cs]
        rot_mat[2,2] = 1
        # Need to rotate around center
        t_mat = np.eye(3)
        t_mat[0,2] = -res[1]/2
        t_mat[1,2] = -res[0]/2
        t_inv = t_mat.copy()
        t_inv[:2,2] *= -1
        t = np.dot(t_inv,np.dot(rot_mat,np.dot(t_mat,t)))
    return t


def transform(pt, center, scale, res, invert=0, rot=0):
    # Transform pixel location to different reference
    t = get_transform(center, scale, res, rot=rot)
    if invert:
        t = np.linalg.inv(t)
    new_pt = np.array([pt[0] - 1, pt[1] - 1, 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2].astype(int) + 1


def transform_preds(coords, center, scale, res):
    # size = coords.size()
    # coords = coords.view(-1, coords.size(-1))
    # print(coords.size())
    for p in range(coords.size(0)):
        coords[p, 0:2] = to_torch(transform(coords[p, 0:2], center, scale, res, 1, 0))
    return coords


def crop_ori(img, center, scale, res, rot=0):
    img = im_to_numpy(img)

    # Preprocessing for efficient cropping
    ht, wd = img.shape[0], img.shape[1]
    sf = scale * 200.0 / res[0]
    if sf < 2:
        sf = 1
    else:
        new_size = int(np.math.floor(max(ht, wd) / sf))
        new_ht = int(np.math.floor(ht / sf))
        new_wd = int(np.math.floor(wd / sf))
        if new_size < 2:
            return torch.zeros(res[0], res[1], img.shape[2]) \
                        if len(img.shape) > 2 else torch.zeros(res[0], res[1])
        else:
            img = scipy.misc.imresize(img, [new_ht, new_wd])
            center = center * 1.0 / sf
            scale = scale / sf

    # Upper left point
    ul = np.array(transform([0, 0], center, scale, res, invert=1))
    # Bottom right point
    br = np.array(transform(res, center, scale, res, invert=1))

    # Padding so that when rotated proper amount of context is included
    pad = int(np.linalg.norm(br - ul) / 2 - float(br[1] - ul[1]) / 2)
    if not rot == 0:
        ul -= pad
        br += pad

    new_shape = [br[1] - ul[1], br[0] - ul[0]]
    if len(img.shape) > 2:
        new_shape += [img.shape[2]]
    new_img = np.zeros(new_shape)

    # Range to fill new array
    new_x = max(0, -ul[0]), min(br[0], img.shape[1]) - ul[0]
    new_y = max(0, -ul[1]), min(br[1], img.shape[0]) - ul[1]
    # Range to sample from original image
    old_x = max(0, ul[0]), min(img.shape[1], br[0])
    old_y = max(0, ul[1]), min(img.shape[0], br[1])
    new_img[new_y[0]:new_y[1], new_x[0]:new_x[1]] = img[old_y[0]:old_y[1], old_x[0]:old_x[1]]

    if not rot == 0:
        # Remove padding
        new_img = scipy.misc.imrotate(new_img, rot)
        new_img = new_img[pad:-pad, pad:-pad]

    # new_img = im_to_torch(scipy.misc.imresize(new_img, res))
    new_img = im_to_torch(cv2.resize(new_img, res))
    return new_img
