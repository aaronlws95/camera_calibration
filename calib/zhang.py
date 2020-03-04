import numpy as np
import calib.dlt as dlt

def vij(H, i, j):
    """
    Define vij vector
    Input:
        H: homography matrix (3, 3)
        i: 0 or 1
        j: 0 or 1
    Output:
        vij vector
    """
    return np.array([
        H[0, i] * H[0, j],
        H[0, i] * H[1, j] + H[1, i] * H[0, j],
        H[1, i] * H[1, j],
        H[2, i] * H[0, j] + H[0, i] * H[2, j],
        H[2, i] * H[1, j] + H[1, i] * H[2, j],
        H[2, i] * H[2, j]
    ])

def get_intrinsics(world_data, image_data):
    """
    Get intrinsic matrix
    Input:
        world_data: real world points (k, n, 2)
        image_data: image_data (k, n, 2)
    Output:
        homographies: homography matrices (k, 3, 3)
        pred_intr_mat: intrinsic matrix (3, 3)
    """
    V = []
    homographies = []
    for wpts, ipts in zip(world_data, image_data):
        H = dlt.estimate_homography(wpts, ipts)
        homographies.append(H)
        V.append(vij(H, 0, 1))
        V.append(vij(H, 0, 0) - vij(H, 1, 1))
    V = np.array(V)
    homographies = np.array(homographies)

    _, _, Vt = np.linalg.svd(V)
    b = Vt[-1]

    w = b[0] * b[2] * b[5] - b[1]**2 * b[5] - b[0] * b[4]**2 + 2 * b[1] * b[3] * b[4] - b[2] * b[3]**2
    d = b[0] * b[2] - b[1]**2

    alpha = np.sqrt(w / (d * b[0]))
    beta = np.sqrt(w / d**2 * b[0])
    gamma = np.sqrt(w / (d**2 * b[0])) * b[1]
    uc = (b[1] * b[4] - b[2] * b[3]) / d
    vc = (b[1] * b[3] - b[0] * b[4]) / d

    pred_intr_mat = np.array([
        [alpha, gamma, uc],
        [0,     beta,  vc],
        [0,     0,      1]
    ])

    return homographies, pred_intr_mat

def get_extrinsics(homographies, pred_intr_mat):
    """
    Get extrinsic matrix
    Input:
        homographies: homographiy matrices (k, 3, 3)
        pred_intr_mat: intrinsic matrix (3, 3)
    Output:
        pred_extr_mat: extrinsic matrix (3, 4)
    """
    pred_extr_mat = []
    inv_intrinsics = np.linalg.inv(pred_intr_mat)
    for homography in homographies:
        h0 = homography[:, 0]
        h1 = homography[:, 1]
        h2 = homography[:, 2]

        ld = 1 / np.linalg.norm(np.dot(inv_intrinsics, h0))

        r0 = ld * np.dot(inv_intrinsics, h0)
        r1 = ld * np.dot(inv_intrinsics, h1)
        r2 = np.cross(r0, r1)

        t = np.array(ld * np.dot(inv_intrinsics, h2)).T

        Rt = np.array([r0.T, r1.T, r2.T, t.T]).T

        pred_extr_mat.append(Rt)

    return pred_extr_mat