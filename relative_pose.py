# import the necessary packages
import argparse
from turtle import shape
import imutils
import numpy as np
import cv2
import sys

firstMarkerID = 1
secondMarkerID = 0

def inversePerspective(rvec, tvec):
    R, _ = cv2.Rodrigues(rvec)
    R = np.matrix(R).T
    invTvec = np.dot(R, np.matrix(-tvec))
    invRvec, _ = cv2.Rodrigues(R)
    return invRvec, invTvec

def relativePosition(rvec1, tvec1, rvec2, tvec2):
    """ Get relative position for rvec2 & tvec2. Compose the returned rvec & tvec to use composeRT with rvec2 & tvec2 """
    rvec1, tvec1 = rvec1.reshape((3, 1)), tvec1.reshape((3, 1))
    rvec2, tvec2 = rvec2.reshape((3, 1)), tvec2.reshape((3, 1))
    # Inverse the second marker
    invRvec, invTvec = inversePerspective(rvec2, tvec2)
    info = cv2.composeRT(rvec1, tvec1, invRvec, invTvec)
    composedRvec, composedTvec = info[0], info[1]
    composedRvec = composedRvec.reshape((3, 1))
    composedTvec = composedTvec.reshape((3, 1))
    return composedRvec, composedTvec

def draw(img, corners, imgpts):
    imgpts = np.int32(imgpts).reshape(-1,2)
    # draw ground floor in green
    # img = cv2.drawContours(img, [imgpts[:4]],-1,(0,255,0),-3)
    # draw pillars in blue color
    for i,j in zip(range(4),range(4)):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]),(255),3)
    # draw top layer in red color
    return img

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
# print(image.shape)
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
markerTvecList = []
markerRvecList = []
composedRvec, composedTvec = None, None
corners, ids, rejected = cv2.aruco.detectMarkers(image, arucoDict, parameters=arucoParams)
camera_matrix = np.array([540.0, 0.0, 1024.0, 0.0, 540.0, 540.0, 0.0, 0.0, 1.0]).reshape(3,3)
camera_dist = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
markerLength = 1.0
if np.all(ids is not None): 
	del markerTvecList[:]
	del markerRvecList[:]
	zipped = zip(ids, corners)
	ids, corners = zip(*(sorted(zipped)))
	axis = np.float32([[-0.01, -0.01, 0], [-0.01, 0.01, 0], [0.01, -0.01, 0], [0.01, 0.01, 0]]).reshape(-1, 3)
	for i in range (0, len(ids)):
		# print(corners[i][0][0])
		# corners[i] = corners[i].reshape((4, 2))
		rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners=corners[i], markerLength = markerLength, cameraMatrix=camera_matrix, distCoeffs=camera_dist)
		
		if ids[i] == firstMarkerID:
			firstRvec = rvec
			firstTvec = tvec
			isFirstMarkerCalibrated = True
			firstMarkerCorners = corners[i]
		elif ids[i] == secondMarkerID:
			secondRvec = rvec
			secondTvec = tvec
			isSecondMarkerCalibrated = True
			secondMarkerCorners = corners[i]
		(rvec - tvec).any() 
		markerRvecList.append(rvec)
		markerTvecList.append(tvec)
		corner = np.squeeze(corners[i], axis=0)
		print(corner.shape)

		corner = corner.reshape((4, 2))
		(topLeft, topRight, bottomRight, bottomLeft) = corner
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
		cv2.putText(image, str(ids[i]),
			(topLeft[0], topLeft[1] - 15), cv2.FONT_HERSHEY_SIMPLEX,
			0.5, (0, 255, 0), 2)
		cv2.aruco.drawDetectedMarkers(image, corners)
		cv2.aruco.drawAxis(image, camera_matrix, camera_dist, rvec, tvec, 2.0)
		cv2.imshow("Image", image)
		cv2.waitKey(0)

	composedRvec, composedTvec = relativePosition(markerRvecList[1], markerTvecList[1], markerRvecList[0], markerTvecList[0])
	print("first: ", markerRvecList[0], markerTvecList[0])  # first marker vectors
	print("second: ", markerRvecList[1], markerTvecList[1])  # second marker vectors
	print("composed: ", composedRvec, composedTvec)  # relative marker vectors

	
