import cv2
import numpy as np
import os
import glob

# Step 1 : Xác định toạ độ thực của các điểm 3D trên bàn cờ kích thước 8x6
# kich thuoc ban co
CHECKERBOARD = (6, 8)

criteria = (cv2.TERM_CRITERIA_EPS +
			cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


# toa do trong khong gian thuc
threedpoints = []

# toa do pixel
twodpoints = []

objectp3d = np.zeros((1, CHECKERBOARD[0]
					* CHECKERBOARD[1],
					3), np.float32)
objectp3d[0, :, :2] = np.mgrid[0:CHECKERBOARD[0],
							0:CHECKERBOARD[1]].T.reshape(-1, 2)
prev_img_shape = None

# Step 2 : Doc file anh (17 anh )
images = glob.glob('*.jpg')
# Step 3 : Tìm các góc của bàn cờ và tinh chỉnh các góc đó
for filename in images:
	image = cv2.imread(filename)
	grayColor = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	ret, corners = cv2.findChessboardCorners(
					grayColor, CHECKERBOARD,
					cv2.CALIB_CB_ADAPTIVE_THRESH
					+ cv2.CALIB_CB_FAST_CHECK +
					cv2.CALIB_CB_NORMALIZE_IMAGE)

	if ret == True:
		threedpoints.append(objectp3d)

		corners2 = cv2.cornerSubPix(
			grayColor, corners, (11, 11), (-1, -1), criteria)

		twodpoints.append(corners2)

		# ve va hien cac goc ban co da tim duoc
		image = cv2.drawChessboardCorners(image,
										CHECKERBOARD,
										corners2, ret)

	cv2.imshow('img',cv2.resize(image, (800, 600)))
	cv2.waitKey(0)
   
cv2.destroyAllWindows()

h, w = image.shape[:2]


# Step 4 : hieu chinh may anh
ret, matrix, distortion, r_vecs, t_vecs = cv2.calibrateCamera(
	threedpoints, twodpoints, grayColor.shape[::-1], None, None)
focal_length = (matrix[0][0]+matrix[1][1])/2000
# print(focal_length)
#Save parameters into numpy file
np.save("./ret", ret)
np.save("./K", matrix)
np.save("./dist", distortion)
np.save("./rvecs", r_vecs)
np.save("./tvecs", t_vecs)
np.save("./FocalLength", focal_length)