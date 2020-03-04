import numpy as np

def world2image(world_pts, proj_mat):
    """
    Project world points to image points with a projection matrix
    Input:
        world_pts: real world points (3, 2) or (3, 3)
        proj_mat: projection matrix (3, 3) or (3, 4)
    Output:
        image_pts: projected image points (3, 2)
    """
    image_pts = np.matmul(proj_mat, np.vstack((world_pts.T, np.ones((1, world_pts.shape[0])))))
    image_pts = image_pts.T
    image_pts = image_pts[:, :2]/image_pts[:, 2][:, np.newaxis]
    return image_pts