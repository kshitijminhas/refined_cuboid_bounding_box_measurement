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
max_len_contours = 50

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
    print('Rescaled bounding box')
    print(box)

    #detect the edges
    low_threshold = 20
    img_blur = cv2.blur(src_gray, (3,3))
    detected_edges = cv2.Canny(img_blur, low_threshold, low_threshold*ratio, kernel_size)
    #create the mask and crop based on the bounding box
    mask = detected_edges != 0
    dst = src_gray * (mask[:, :].astype(src.dtype))
    # cv2.namedWindow('test')
    cv2.imshow('canny_mask', dst)
    dst=dst[box[1]:box[3],box[0]:box[2]]
    cv2.imshow('cropped_canny_mask', dst)

    #get rid of "noisy" edges that aren't a part of the main object structure
    ret, thresh = cv2.threshold(dst, 50, 255, cv2.THRESH_BINARY)
    # cv2.imshow('threshold', thresh)
    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cv2.imshow('contours', im2)
    mask = np.zeros_like(dst)
    contours=[c for c in contours if len(c)>max_len_contours]
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
    cv2.imshow('corner_blobs', dst)
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)

    # define the criteria to stop and refine the corners
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv2.cornerSubPix(mask, np.float32(centroids), (5, 5), (-1, -1), criteria)
    # res = np.hstack((centroids, corners))
    res = np.int0(centroids)
    # import pdb; pdb.set_trace()
    res+=box[:2]
    return res

def best_fit_transform(A, B):
    '''
    Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
    Input:
      A: Nxm numpy array of corresponding points
      B: Nxm numpy array of corresponding points
    Returns:
      T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
      R: mxm rotation matrix
      t: mx1 translation vector
    '''

    assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    # rotation matrix
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
       Vt[m-1,:] *= -1
       R = np.dot(Vt.T, U.T)

    # translation
    t = centroid_B.T - np.dot(R,centroid_A.T)

    # homogeneous transformation
    T = np.identity(m+1)
    T[:m, :m] = R
    T[:m, m] = t

    return T, R, t

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Code for Refined Bounding Boxes')
    parser.add_argument("--image", type=str, default="images/00170_colors.png", help="path to dataset")
    parser.add_argument("--image_right", type=str, default="00171_colors.png", help="path to dataset")
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
    print('Bounding box from YOLO')
    print(boxes[0][0])

    masks=[]
    corners=[]
    src = cv2.imread(args.image)
    cv2.imshow('orig', src)
    src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('orig_gray', src_gray)
    #only taking the first yolo bounding box, TBD: how to select the best bbox?
    masks.append(CannyThreshold(boxes[0][0], src_gray, args))
    corners.append(getCorners(masks[-1],boxes[0][0]))
    print('press any key on opencv window to continue')
    cv2.waitKey(0)

    # # extract the edge masks from the parts of the image inside the bounding boxes
    # for batch_i, (img_paths, input_imgs) in enumerate(dataloader):
    #     src = cv2.imread(img_paths[0])
    #     src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    #     for box in boxes[batch_i]:
    #         # import pdb; pdb.set_trace()
    #         print(box)
    #         masks.append(CannyThreshold(box, src_gray, args))
    #         corners.append(getCorners(masks[-1],box))

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

    inputs_right = load_images( glob.glob(args.image_right) )
    print('\nLoaded ({0}) images of size {1}.'.format(inputs_right.shape[0], inputs_right.shape[1:]))

    # Compute results
    outputs = predict(model, inputs)
    print('left output shape')
    print(outputs.shape[:])

    outputs_right = predict(model, inputs_right)
    print('right output shape')
    print(outputs_right.shape[:])

    # Display DenseDepth results
    viz = display_images(outputs.copy(), inputs.copy())
    plt.figure(figsize=(10,5))
    plt.imshow(viz)

    viz_right = display_images(outputs_right.copy(), inputs_right.copy())
    plt.figure(figsize=(10,5))
    plt.imshow(viz_right)

    # Following code takes in corner points and displays a 3d plot with their densedepth estimates
    # for point in corner_points_from_file:
    corner_points_with_depth=[]
    for c in corners:
        # print('all corners in bounding box with depth')
        for point in c:
            corner_points_with_depth.append([point[0], point[1], outputs[0, math.ceil(point[1] / 2), math.ceil(point[0] / 2), 0] * 10])
        # print(corner_points_with_depth)

    plt.figure(2)
    ax = plt.axes(projection = '3d')
    for point in corner_points_with_depth:
        ax.scatter3D(point[0], point[2], point[1])

    ax.set_xlabel('u')
    ax.set_ylabel('z')
    ax.set_zlabel('v')

    # NYUv2 - Kinect - RGB Intrinsic Parameters
    fx = 5.1885790117450188e+02
    fy = 5.1946961112127485e+02
    cx = 3.2558244941119034e+02
    cy = 2.5373616633400465e+02

    K = np.array([[fx,   0,  cx],
         [0,    fy, cy],
         [0,    0,   1]])

    plt.figure(3)
    ay = plt.axes(projection = '3d')

    for point in corner_points_with_depth:
        z = point[2]
        uv = np.array([[point[0]],
                        [point[1]],
                        [1]])
        world_point = np.matmul(np.linalg.inv(K), z*uv)
        ay.scatter3D(world_point[0], world_point[2], world_point[1])
        ay.invert_zaxis()
        # print(world_point)

    ay.set_xlabel('x')
    ay.set_zlabel('y')
    ay.set_ylabel('z')

    image_left = cv2.imread('images/00170_colors.png')
    image_right = cv2.imread('00171_colors.png')

    sift_left = cv2.SIFT_create()
    sift_right = cv2.SIFT_create()

    kp_left, des_left = sift_left.detectAndCompute(image_left, None)
    kp_right, des_right = sift_right.detectAndCompute(image_right, None)
    
    kp_left = np.float32([i.pt for i in kp_left])
    kp_right = np.float32([i.pt for i in kp_right])

    match_instance = cv2.DescriptorMatcher_create("BruteForce")
    All_Matches = match_instance.knnMatch(des_left, des_right, 2)

    valid_matches = []
    lowe_ratio = 0.8
    i = 0

    for val in All_Matches:
        if len(val) == 2 and val[0].distance < val[1].distance * lowe_ratio:
            valid_matches.append((val[0].trainIdx, val[0].queryIdx))

    if len(valid_matches) > 4:
    # construct the two sets of points
        print('more than 4 valid matches found')
        pointsA = np.float32([kp_left[i] for (_,i) in valid_matches])
        pointsB = np.float32([kp_right[i] for (i,_) in valid_matches])
    
    sift_points_with_depth=[]
    for point in pointsA:
        sift_points_with_depth.append([point[0], point[1], outputs[0, math.ceil(point[1] / 2), math.ceil(point[0] / 2), 0] * 10])

    sift_points_with_depth_right=[]
    for point in pointsB:
        sift_points_with_depth_right.append([point[0], point[1], outputs_right[0, math.ceil(point[1] / 2), math.ceil(point[0] / 2), 0] * 10])

    sift_points_in_world=np.empty((0,3), float)
    for point in sift_points_with_depth:
        z = point[2]
        uv = np.array([[point[0]],
                        [point[1]],
                        [1]])
        sift_points_in_world = np.append(sift_points_in_world,np.transpose(np.matmul(np.linalg.inv(K), z*uv)),axis=0)


    sift_points_in_world_right=np.empty((0,3), float)
    for point in sift_points_with_depth_right:
        z = point[2]
        uv = np.array([[point[0]],
                        [point[1]],
                        [1]])
        sift_points_in_world_right = np.append(sift_points_in_world_right,np.transpose(np.matmul(np.linalg.inv(K), z*uv)),axis=0)
    
    print('sift_points_in_world')
    # print(sift_points_in_world)
    print(sift_points_in_world.shape[:])

    print('sift_points_in_world_right')
    # print(sift_points_in_world_right)
    print(sift_points_in_world_right.shape[:])

    T, R, t = best_fit_transform(sift_points_in_world, sift_points_in_world_right)
    print(T)
    print(R)
    print(t)

    plt.figure(5)
    a1 = plt.axes(projection = '3d')
    a1.invert_zaxis()
    # import pdb
    # pdb.set_trace()
    
    for world_point in sift_points_in_world:
        trans_p = np.dot(R, np.transpose(world_point)) + t
        a1.scatter3D(trans_p[0], trans_p[2], trans_p[1], c='b', alpha=0.5)
    
    for world_point in sift_points_in_world_right:
        a1.scatter3D(world_point[0], world_point[2], world_point[1], c='r', alpha=0.5)

    a1.set_xlabel('x')
    a1.set_zlabel('y')
    a1.set_ylabel('z')
    print('press q on all matplotlib windows to quit program')
    plt.show()