import subprocess
import yaml
import rosbag
import cv2
from cv_bridge import CvBridge
import numpy as np


FILENAME = '_2022-08-19-16-51-16'
ROOT_DIR = '/home/ducanh/ducanh_ws/bag_record/fly_bag'
BAGFILE = ROOT_DIR + '/' + FILENAME + '.bag'

if __name__ == '__main__':
    bag = rosbag.Bag(BAGFILE)
    for i in range(2):
        if (i == 0):
            TOPIC = '/camera/depth/image_rect_raw'
            DESCRIPTION = 'depth_'
        else:
            TOPIC = '/camera/color/image_raw'
            DESCRIPTION = 'color_'
        image_topic = bag.read_messages(TOPIC)
        for k, b in enumerate(image_topic):
            bridge = CvBridge()
            cv_image = bridge.imgmsg_to_cv2(b.message, b.message.encoding)
            cv_image.astype(np.uint8)
            if (DESCRIPTION == 'depth_'):
                depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(cv_image, alpha=0.03), cv2.COLORMAP_JET)
                cv2.imwrite(ROOT_DIR + '/depth/' + DESCRIPTION + str(b.timestamp) + '.png', cv_image)
            else:
                cv2.imwrite(ROOT_DIR + '/color/' + DESCRIPTION + str(b.timestamp) + '.png', cv_image)
            print('saved: ' + DESCRIPTION + str(b.timestamp) + '.png')


    bag.close()

    print('PROCESS COMPLETE')