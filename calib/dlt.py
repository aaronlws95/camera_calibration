import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def normalization(data):
    """
    Normalize data input (centroid to origin and mean distance of sqrt 2 or 3)
    Input:
        data: input data (n, 2) or (n, 3)
    Output:
        tr: normalization transform matrix (3, 3) or (3, 4)
        norm_data: normalized data (n, 2) or (n, 3)
    """
    assert data.shape[1] == 2 or data.shape[1] == 3

    m = np.mean(data, 0)
    s = np.sqrt(2) / np.std(data) if data.shape[1] == 2 else np.sqrt(3) / np.std(data)

    Tr = np.array([[s, 0, -s * m[0]], [0, s, -s * m[1]], [0, 0, 1]]) if data.shape[1] == 2 else \
        np.array([[s, 0, 0, -s * m[0]], [0, s, 0, -s * m[1]], [0, 0, s, -s * m[2]], [0, 0, 0, 1]])

    norm_data = np.dot(Tr, np.vstack((data.T, np.ones((1, data.shape[0]))))).T

    return Tr, norm_data

def estimate_homography(world_data, image_data):
    """
    Estimate homography (Ah = 0) matrix H with Direct Linear Transform
    Input:
        world_data: real world point (n, 2) or (n, 3)
        image_data: image points (n, 2)
    Output:
        H: homography matrix (3, 3) or (3, 4)
    """
    nd = world_data.shape[1]

    world_tr, world_norm = normalization(world_data)
    image_tr, image_norm = normalization(image_data)

    A = []

    for ipt, wpt in zip(image_norm, world_norm):

        a1 = [wpt[0], wpt[1], 1, 0, 0, 0, -ipt[0]*wpt[0], -ipt[0]*wpt[1], -ipt[0]] if nd == 2 else \
            [wpt[0], wpt[1], wpt[2], 1, 0, 0, 0, 0, -ipt[0]*wpt[0], -ipt[0]*wpt[1], -ipt[0]*wpt[2], -ipt[0]]
        a2 = [0, 0, 0, wpt[0], wpt[1], 1, -ipt[1]*wpt[0], -ipt[1]*wpt[1], -ipt[1]] if nd == 2 else \
            [0, 0, 0, 0, wpt[0], wpt[1], wpt[2], 1, -ipt[1]*wpt[0], -ipt[1]*wpt[1], -ipt[1]*wpt[2], -ipt[1]]

        A.append(a1)
        A.append(a2)

    _, _, Vt = np.linalg.svd(np.array(A))

    L = Vt[-1]

    H = L.reshape(3, nd + 1)

    H = np.dot(np.dot(np.linalg.inv(image_tr), H), world_tr)

    return H / H[-1, -1]

