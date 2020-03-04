import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from calib.utils import world2image
import scipy.optimize

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

    # H = refine_homography(H, world_data, image_data)

    return H / H[-1, -1]

def error_func(homography, world_data, image_data):
    """
    Optimization error function
    Input:
        homography: homography matrix (3, 3) or (3, 4)
        world_data: real world data (n, 2) or (n, 3)
        image_data: observed image data (n, 2)
    Output:
        error: error value to minimize (2n)
    """
    nd = world_data.shape[1]
    homography = homography.reshape(3, nd + 1)

    pred_image_data = world2image(world_data, homography)
    error = np.square(pred_image_data.flatten() - image_data.flatten())
    return error

def jacobian(homography, world_data, image_data):
    """
    Optimization jacobian matrix
    Input:
        homography: homography matrix (3, 3) or (3, 4)
        world_data: real world data (n, 2) or (n, 3)
        image_data: observed image data (n, 2)
    Output:
        J: jacobian matrix (2n, 9) or (2n, 12)
    """
    nd = world_data.shape[1]
    homography = homography.reshape(3, nd + 1)

    J = []
    for d in world_data:

        cur_d = np.vstack((d.reshape(nd, 1), 1))
        sw = np.matmul(homography, cur_d)

        sx = sw[0, 0]
        sy = sw[1, 0]
        w = sw[2, 0]

        j1 = [d[0] / w, d[1] / w, 1 / w, 0, 0, 0, (-sx * d[0]) / w*w, (-sx * d[1]) / w*w, -sx / w*w] if nd == 2 else \
            [d[0] / w, d[1] / w, d[2] / w, 1 / w, 0, 0, 0, 0, (-sx * d[0]) / w*w, (-sx * d[1]) / w*w, (-sx * d[2]) / w*w, -sx / w*w]

        j2 = [0, 0, 0, d[0] / w, d[1] / w, 1 / w, (-sy * d[0]) / w*w, (-sy * d[1]) / w*w, -sy / w*w] if nd == 2 else \
            [0, 0, 0, 0, d[0] / w, d[1] / w, d[2] / w, 1 / w, (-sy * d[0]) / w*w, (-sy * d[1]) / w*w, (-sy * d[2]) / w*w, -sy / w*w]

        J.append(j1)
        J.append(j2)

    return np.array(J)

def refine_homography(homography, world_data, image_data, jac=jacobian, cost_fn=error_func):
    """
    Refine homography matrix with Levenberg-Marquardt optimization algorithm
    Input:
        homography: homography matrix (3, 3) or (3, 4)
        world_data: real world data (n, 2) or (n, 3)
        image_data: observed image data (n, 2)
        jac: function to return jacobian matrix
        cost_fn: function to return error
    Output:
        refined: refined homography matrix (3, 3) or (3, 4)
    """

    refined = scipy.optimize.least_squares(fun=cost_fn, x0=homography.flatten(), jac=jac, args=[world_data, image_data], method='lm').x
    refined = refined.reshape((3, world_data.shape[1] + 1))
    refined = refined/refined[-1, -1]
    return refined