# import the necessary packages
import argparse
import imutils
import numpy as np
import cv2
import sys

#from torch import double
#import cv2.aruco as aruco
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image containing ArUCo tag")
ap.add_argument("-t", "--type", type=str,
	default="DICT_ARUCO_ORIGINAL",
	help="type of ArUCo tag to detect")
args = vars(ap.parse_args())
# define names of each possible ArUco tag OpenCV supports
ARUCO_DICT = {
	"DICT_4X4_50": cv2.aruco.DICT_4X4_50,
	"DICT_4X4_100": cv2.aruco.DICT_4X4_100,
	"DICT_4X4_250": cv2.aruco.DICT_4X4_250,
	"DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
	"DICT_5X5_50": cv2.aruco.DICT_5X5_50,
	"DICT_5X5_100": cv2.aruco.DICT_5X5_100,
	"DICT_5X5_250": cv2.aruco.DICT_5X5_250,
	"DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
	"DICT_6X6_50": cv2.aruco.DICT_6X6_50,
	"DICT_6X6_100": cv2.aruco.DICT_6X6_100,
	"DICT_6X6_250": cv2.aruco.DICT_6X6_250,
	"DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
	"DICT_7X7_50": cv2.aruco.DICT_7X7_50,
	"DICT_7X7_100": cv2.aruco.DICT_7X7_100,
	"DICT_7X7_250": cv2.aruco.DICT_7X7_250,
	"DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
	"DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
	"DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
	"DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
	"DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
	"DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
}

# load the input image from disk and resize it
print("[INFO] loading image...")
image = cv2.imread(args["image"])
# image = imutils.resize(image, width=6000, height=4000)
# verify that the supplied ArUCo tag exists and is supported by
# OpenCV
if ARUCO_DICT.get(args["type"], None) is None:
	print(f"ArUCo tag type '{args['type']}' is not supported")
	sys.exit(0)
# load the ArUCo dictionary, grab the ArUCo parameters, and detect
# the markers
print("[INFO] detecting '{}' tags...".format(args["type"]))
arucoDict = cv2.aruco.Dictionary_get(ARUCO_DICT[args["type"]])
arucoParams = cv2.aruco.DetectorParameters_create()
# arucoParams.adaptiveThreshConstant = 7
corners, ids, rejected = cv2.aruco.detectMarkers(image, arucoDict, parameters=arucoParams)
camera_matrix = np.array([1080.0, 0.0, 2048.0, 0.0, 1080.0, 1080.0, 0.0, 0.0, 1.0]).reshape(3,3)
camera_dist = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
markerLength = 2.0
print(len(corners))
# verify *at least* one ArUco marker was detected
if len(corners) > 0:
	# flatten the ArUco IDs list
	ids = ids.flatten()
	# for i in range (0, len(ids)):
		# loop over the detected ArUCo corners
	for (markerCorner, markerID) in zip(corners, ids):
		# extract the marker corners (which are always returned in
		# top-left, top-right, bottom-right, and bottom-left order)
		ret1 = cv2.aruco.estimatePoseSingleMarkers(corners=markerCorner, markerLength = markerLength, cameraMatrix=camera_matrix, distCoeffs=camera_dist)
		# print(ret1)
		rvec, tvec = ret1[0][0, 0, :], ret1[1][0, 0, :]
		rmat,_ = cv2.Rodrigues(rvec)
		rmat = np.array(rmat, dtype=np.float32)
		tvec_inv = np.matmul(-rmat.T, tvec)
		# print(tvec_inv)
		# -- Draw the detected marker and put a reference frame over it
		# cv2.aruco.drawDetectedMarkers(image, corners, ids)
		cv2.aruco.drawAxis(image, camera_matrix, camera_dist, rvec, tvec, 0.2)
		corners = markerCorner.reshape((4, 2))
		(topLeft, topRight, bottomRight, bottomLeft) = corners
		# convert each of the (x, y)-coordinate pairs to integers
		topRight = (int(topRight[0]), int(topRight[1]))
		bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
		bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
		topLeft = (int(topLeft[0]), int(topLeft[1]))
				# draw the bounding box of the ArUCo detection
		cv2.line(image, topLeft, topRight, (0, 255, 0), 2)
		cv2.line(image, topRight, bottomRight, (0, 255, 0), 2)
		cv2.line(image, bottomRight, bottomLeft, (0, 255, 0), 2)
		cv2.line(image, bottomLeft, topLeft, (0, 255, 0), 2)
		# compute and draw the center (x, y)-coordinates of the ArUco
		# marker
		cX = int((topLeft[0] + bottomRight[0]) / 2.0)
		cY = int((topLeft[1] + bottomRight[1]) / 2.0)
		cv2.circle(image, (cX, cY), 4, (0, 0, 255), -1)
		# draw the ArUco marker ID on the image
		cv2.putText(image, str(markerID),
			(topLeft[0], topLeft[1] - 15), cv2.FONT_HERSHEY_SIMPLEX,
			0.5, (0, 255, 0), 2)
		print("Position camera in marker frame: ", tvec_inv)
		print("[INFO] ArUco marker ID: {}".format(markerID))
		print("At the position ", cX, cY)
		# show the output image
		cv2.imshow("Image", image)
		cv2.waitKey(0)
