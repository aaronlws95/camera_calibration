import numpy as np
import cv2
import glob
import calib.dlt as dlt
import calib.zhang as zhang
from calib.utils import world2image
import matplotlib.pyplot as plt

def draw_points(ax, pts, c=None):
    x = pts[:, 0]
    y = pts[:, 1]
    ax.scatter(x, y, 2, 'r')

# Load data
wpt = np.zeros((6*7, 2), np.float32)
wpt[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)

world_pts = []
image_pts = []

image_files = glob.glob('data/opencv_chessboard/*.jpg')
images = []
for img_f in image_files:
    img = cv2.imread(img_f)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(img, (7,6), None)

    if ret == True:
        world_pts.append(wpt)
        ipt = cv2.cornerSubPix(img, corners, (11, 11), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)).squeeze()
        image_pts.append(ipt)
        images.append(img_f)

        # fig, ax = plt.subplots()
        # ax.imshow(img)
        # draw_points(ax, ipt)
        # plt.show()

world_pts = np.asarray(world_pts)
image_pts = np.asarray(image_pts)

# Camera Calibration
homographies, pred_intr_mat = zhang.get_intrinsics(world_pts, image_pts)
pred_extr_mat = zhang.get_extrinsics(homographies, pred_intr_mat)

# Reconstruction error
rec_error = 0
for i in range(len(world_pts)):
    pred_proj_mat = np.matmul(pred_intr_mat, pred_extr_mat[i])
    pred_image_pts = world2image(np.hstack((world_pts[i], np.zeros((world_pts[i].shape[0], 1)))), pred_proj_mat)
    cur_image_pts = image_pts[i].squeeze()
    cur_error = np.square(pred_image_pts - cur_image_pts)
    cur_error = np.sqrt(cur_error[:, 0] + cur_error[:, 1])
    cur_error = sum(cur_error)/len(cur_error)
    rec_error += cur_error

    # Plot
    # img = cv2.imread(images[i])
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # fig, ax = plt.subplots(1, 2)
    # ax[0].imshow(img)
    # ax[0].set_title('ori')
    # draw_points(ax[0], cur_image_pts)
    # ax[1].imshow(img)
    # ax[1].set_title('pred')
    # draw_points(ax[1], pred_image_pts)
    # plt.show()

rec_error = rec_error/len(world_pts)
print('Reconstruction Error: %0.12f' %rec_error)