import numpy as np
import cv2
import glob

# Load data
wpt = np.zeros((6*7, 3), np.float32)
wpt[:,:2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)
world_pts = []
image_pts = []
images = glob.glob('data/opencv_chessboard/*.jpg')

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, (7 ,6), None)

    if ret == True:
        world_pts.append(wpt)

        ipt = cv2.cornerSubPix(gray, corners,(11, 11), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        image_pts.append(ipt)

#         img = cv2.drawChessboardCorners(img, (7, 6), ipt, ret)
#         cv2.imshow('img', img)
#         cv2.waitKey(500)
# cv2.destroyAllWindows()

# Calibration
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(world_pts, image_pts, gray.shape[::-1],None,None)

# Reconstruction error
rec_error = 0
for i in range(len(world_pts)):
    pred_image_pts, _ = cv2.projectPoints(world_pts[i], rvecs[i], tvecs[i], mtx, dist)
    pred_image_pts = pred_image_pts.squeeze()
    cur_image_pts = image_pts[i].squeeze()
    cur_error = np.square(pred_image_pts - cur_image_pts)
    cur_error = np.sqrt(cur_error[:, 0] + cur_error[:, 1])
    cur_error = sum(cur_error)/len(cur_error)
    rec_error += cur_error
rec_error = rec_error/len(world_pts)
print('Reconstruction Error: %0.12f' %rec_error)