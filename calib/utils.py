import numpy as np

def world2image(world_pts, proj_mat):
    image_pts = np.matmul(proj_mat, np.vstack((world_pts.T, np.ones((1, world_pts.shape[0])))))
    image_pts = image_pts.T
    image_pts = image_pts[:, :2]/image_pts[:, 2][:, np.newaxis]
    return image_pts