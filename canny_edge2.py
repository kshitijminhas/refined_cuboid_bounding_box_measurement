# https://docs.opencv.org/3.4/da/d5c/tutorial_canny_detector.html
from __future__ import print_function
import cv2 as cv
import argparse
max_lowThreshold = 100
window_name = 'Edge Map'
title_trackbar = 'Min Threshold:'
ratio = 3
kernel_size = 3
bounding_box = ()
def CannyThreshold(val):
    low_threshold = val
    img_blur = cv.blur(src_gray, (3,3))
    detected_edges = cv.Canny(img_blur, low_threshold, low_threshold*ratio, kernel_size)
    mask = detected_edges != 0
    # dst = src[bounding_box[0]:bounding_box[1],:,:] * (mask[bounding_box[0]:bounding_box[1],:,None].astype(src.dtype))
    dst = src * (mask[:,:,None].astype(src.dtype))
    cv.imshow(window_name, dst)
parser = argparse.ArgumentParser(description='Code for Canny Edge Detector tutorial.')
parser.add_argument('--image', help='Path to input image.', default='chair1.jpg')
parser.add_argument('--bbox', help='Path to 2D bbox.', default='2d_box1.txt')
args = parser.parse_args()
src = cv.imread(cv.samples.findFile(args.image))
if src is None:
    print('Could not open or find the image: ', args.image)
    exit(0)
bbox = open(args.bbox, 'r')
bounding_box_values = bbox.readlines()
bounding_box = (int(bounding_box_values[0]), int(bounding_box_values[1]))
src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
cv.namedWindow(window_name)
cv.createTrackbar(title_trackbar, window_name , 0, max_lowThreshold, CannyThreshold)
CannyThreshold(0)
cv.waitKey()