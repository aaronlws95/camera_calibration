import numpy as np
import matplotlib.pyplot as plt
import calib.dlt as dlt
import cv2
from calib.utils import world2image

def draw_joints(ax, joints, c=None):
    def _draw2djoints(ax, joints, links, c=None):
        if c:
            colors = [c, c, c, c, c]
        else:
            colors = ['r', 'm', 'b', 'c', 'g']
        for finger_idx, finger_links in enumerate(links):
            for idx in range(len(finger_links) - 1):
                _draw2dseg(
                    ax,
                    joints,
                    finger_links[idx],
                    finger_links[idx + 1],
                    c=colors[finger_idx])

    def _draw2dseg(ax, annot, idx1, idx2, c=None):
        ax.plot([annot[idx1, 0], annot[idx2, 0]],
                [annot[idx1, 1], annot[idx2, 1]], c=c)

    links = [(0, 1, 6, 7, 8), (0, 2, 9, 10, 11), (0, 3, 12, 13, 14),
             (0, 4, 15, 16, 17), (0, 5, 18, 19, 20)]
    x = joints[:, 0]
    y = joints[:, 1]
    ax.scatter(x, y, 2, 'r')
    _draw2djoints(ax, joints, links, c=c)

# Given camera properties
cam_extr = [[0.999988496304, -0.00468848412856, 0.000982563360594, 25.7],
            [0.00469115935266, 0.999985218048, -0.00273845880292, 1.22],
            [-0.000969709653873, 0.00274303671904, 0.99999576807, 3.902]]
cam_extr = np.asarray(cam_extr)

focal_x = 1395.749023
focal_y = 1395.749268
x0 = 935.732544
y0 = 540.681030

cam_intr = [[focal_x, 0, x0],
            [0, focal_y, y0],
            [0, 0, 1]]
cam_intr = np.asarray(cam_intr)

proj_mat = np.matmul(cam_intr, cam_extr)

# Load dataset
f = open('data/FPHA/skeleton.txt', 'r')
img = cv2.imread('data/FPHA/color_0000.jpeg')[:, :, ::-1]
world_pts = f.readlines()[0].strip().split(' ')[1:]
world_pts = np.asarray([float(x) for x in (world_pts)]).reshape(21, 3)
image_pts = world2image(world_pts, proj_mat)

# Camera calibration
pred_proj_mat = dlt.estimate_homography(world_pts, image_pts)

# Reconstruction error
pred_image_pts = world2image(world_pts, pred_proj_mat)
rec_error = np.square(pred_image_pts - image_pts)
rec_error = np.sqrt(rec_error[:, 0] + rec_error[:, 1])
rec_error = sum(rec_error)/len(rec_error)
print('Reconstruction Error: %0.12f' %rec_error)

# Plot
fig, ax = plt.subplots(1, 2)
ax[0].imshow(img)
ax[0].set_title('ori')
draw_joints(ax[0], image_pts)
ax[1].imshow(img)
ax[1].set_title('pred')
draw_joints(ax[1], pred_image_pts)
plt.show()

