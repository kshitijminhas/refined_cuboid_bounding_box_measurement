import sys
sys.path.append('../third_party/PyTorch-YOLOv3/')

from models import *
from utils.utils import *
from utils.datasets import *

import cv2
import argparse
import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# imports for DenseDepth
sys.path.append('../third_party/DenseDepth/')
import os
import glob
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'
from keras.models import load_model
from layers import BilinearUpSampling2D
from densedepth_utils import predict, load_images, display_images
import math
import copy

corner_points = []

def getBoundingBoxes(dataloader, opt):
    imgs = []  # Stores image paths
    img_detections = []  # Stores detections for each image index
    boxes=[]
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    for batch_i, (img_paths, input_imgs) in enumerate(dataloader):
        # Configure input
        input_imgs = Variable(input_imgs.type(Tensor))

        # Get detections
        with torch.no_grad():
            detections = model(input_imgs)
            detections = non_max_suppression(detections, opt.yolo_conf_thres, opt.yolo_nms_thres)

        # Save image and detections
        imgs.extend(img_paths)
        img_detections.extend(detections)
        boxes.extend([[i[:4].int().tolist() for i in detections[0]]])

    return boxes

def CannyThreshold(box, src_gray, args, ratio = 3, kernel_size = 3):
    # unwarp the bounding box coordinates
    box = rescale_boxes(np.array([box]), args.img_size, src_gray.shape)[0]

    #detect the edges
    low_threshold = 20
    img_blur = cv2.blur(src_gray, (3,3))
    detected_edges = cv2.Canny(img_blur, low_threshold, low_threshold*ratio, kernel_size)
    #create the mask and crop based on the bounding box
    mask = detected_edges != 0
    dst = src_gray * (mask[:, :].astype(src.dtype))
    dst=dst[box[1]:box[3],box[0]:box[2]]

    #get rid of "noisy" edges that aren't a part of the main object structure
    ret, thresh = cv2.threshold(dst, 50, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    mask = np.zeros_like(dst)
    contours=[c for c in contours if len(c)>50]
    for temp in contours:
        temp = np.array(temp)
        temp = temp.reshape(len(temp), 2)
        for x,y in temp:
            mask[y,x]=1

    dst=dst*mask
    return dst

def getCorners(mask, box):
    # blur and detect corners
    blur_kernel = (6, 6)
    dst = cv2.cornerHarris(cv2.blur(mask, blur_kernel), 10, 3, 0.04)

    # find centroids
    dst = cv2.dilate(dst, None)
    ret, dst = cv2.threshold(dst, 0.01 * dst.max(), 255, 0)
    dst = np.uint8(dst)
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)

    # define the criteria to stop and refine the corners
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv2.cornerSubPix(mask, np.float32(centroids), (5, 5), (-1, -1), criteria)
    # res = np.hstack((centroids, corners))
    res = np.int0(centroids)
    # import pdb; pdb.set_trace()
    res+=box[:2]
    return res

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Code for Refined Bounding Boxes')
    parser.add_argument("--image", type=str, default="images/00170_colors.png", help="path to dataset")
    parser.add_argument("--image_folder", type=str, default="images", help="path to dataset")
    #TBD: throw error if image does not exist
    parser.add_argument("--yolo_model_def", type=str, default="../third_party/PyTorch-YOLOv3/config/yolov3.cfg",
                        help="path to model definition file")
    parser.add_argument("--yolo_weights_path", type=str, default="../third_party/PyTorch-YOLOv3/weights/yolov3.weights",
                        help="path to weights file")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--yolo_batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--yolo_n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument("--yolo_conf_thres", type=float, default=0.8, help="object confidence threshold")
    parser.add_argument("--yolo_nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
    parser.add_argument('--densedepth_model', default='../third_party/DenseDepth/nyu.h5', type=str, help='Trained Keras model file.')
    args = parser.parse_args()


    # Set up model and device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Darknet(args.yolo_model_def, img_size=args.img_size).to(device)
    model.load_darknet_weights(args.yolo_weights_path)
    model.eval()

    # make a dataloader for the image folder
    dataloader = DataLoader(
            ImageFolder(args.image_folder, img_size=args.img_size),
            batch_size=args.yolo_batch_size,
            shuffle=False,
            num_workers=args.yolo_n_cpu,
        )
    # get the bounding boxes from yolo
    boxes = getBoundingBoxes(dataloader, args)

    # extract the edge masks from the parts of the image inside the bounding boxes
    masks=[]
    corners=[]
    for batch_i, (img_paths, input_imgs) in enumerate(dataloader):
        src = cv2.imread(img_paths[0])
        src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        for box in boxes[batch_i]:
            # import pdb; pdb.set_trace()
            print(box)
            masks.append(CannyThreshold(box, src_gray, args))
            corners.append(getCorners(masks[-1],box))

    # Code to get depth predictions from densedepth
    # Custom object needed for inference and training
    custom_objects = {'BilinearUpSampling2D': BilinearUpSampling2D, 'depth_loss_function': None}
    print('Loading DenseDepth model...')

    # Load model into GPU / CPU
    model = load_model(args.densedepth_model, custom_objects=custom_objects, compile=False)
    print('\nDenseDepth Model loaded ({0}).'.format(args.densedepth_model))

    # Input images
    inputs = load_images( glob.glob(args.image) )
    print('\nLoaded ({0}) images of size {1}.'.format(inputs.shape[0], inputs.shape[1:]))

    # Compute results
    outputs = predict(model, inputs)

    # Display DenseDepth results
    viz = display_images(outputs.copy(), inputs.copy())
    # viz = display_images(outputs.copy())
    plt.figure(figsize=(10,5))
    plt.imshow(viz)
    # plt.savefig('test.png')
    plt.show()

    #TBD: integrate the following code
    # Following code takes in corner points and displays a 3d plot with their densedepth estimates
    # for point in corner_points_from_file:
    for c in corners:
        for points in c:
            import pdb; pdb.set_trace()
            uv = points.split()
    # vu =copy.deepcopy(uv)
    # vu[0] = 240 - math.ceil(float(uv[1]) / 2)
    # vu[1] = math.ceil(float(uv[0]) / 2)
    # corner_points.append(vu)

    # for point in corner_points:
    # point.append(outputs[0, point[0], point[1], 0] * 1000)
    # print(corner_points)

    # fig = plt.figure()
    # ax = plt.axes(projection = '3d')
    # for point in corner_points:
    #     ax.scatter3D(point[1], point[2], point[0])

    # ax.set_xlabel('v')
    # ax.set_ylabel('z')
    # ax.set_zlabel('u')
    # ax.set_xlim([0,320])
    # # ax.set_ylim([0,1000])
    # ax.set_zlim([0,240])
    # plt.show()

    # for i,m in enumerate(masks):
    #     temp=m
    #     #plot the corners
    #     res=corners[i]
    #     temp[res[:, 1], res[:, 0]] = 255
    #     # temp[res[:, 1]+1, res[:, 0]+1] = 255
    #     # temp[res[:, 1], res[:, 0] + 1] = 255
    #     # temp[res[:, 1] + 1, res[:, 0]] = 255
    #     temp[res[:, 1]-1, res[:, 0]-1] = 255
    #     temp[res[:, 1], res[:, 0] - 1] = 255
    #     temp[res[:, 1] - 1, res[:, 0]] = 255
    #     temp[res[:, 3], res[:, 2]] = 255
    #     # temp[res[:, 3]+1, res[:, 2]+1] = 255
    #     # temp[res[:, 3], res[:, 2] + 1] = 255
    #     # temp[res[:, 3] + 1, res[:, 2]] = 255
    #     temp[res[:, 3]-1, res[:, 2]-1] = 255
    #     temp[res[:, 3], res[:, 2] - 1] = 255
    #     temp[res[:, 3] - 1, res[:, 2]] = 255
    #     cv2.imshow('',temp)
    #     cv2.waitKey()

