import sys
sys.path.append('../PyTorch-YOLOv3/') #https://github.com/eriklindernoren/PyTorch-YOLOv3

from models import *
from utils.utils import *
from utils.datasets import *

import cv2
import argparse
import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable


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
            # import pdb; pdb.set_trace()
            detections = model(input_imgs)
            detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)

        # Save image and detections
        imgs.extend(img_paths)
        img_detections.extend(detections)
        boxes.extend([[i[:4].int().tolist() for i in detections[0]]])
        # import pdb; pdb.set_trace()

    return boxes

def CannyThreshold(box, src_gray, args, ratio = 3, kernel_size = 3):
    # import pdb; pdb.set_trace()
    box = rescale_boxes(np.array([box]), args.img_size, src_gray.shape)[0]
    low_threshold = 100
    img_blur = cv2.blur(src_gray, (3,3))
    detected_edges = cv2.Canny(img_blur, low_threshold, low_threshold*ratio, kernel_size)
    mask = detected_edges != 0
    dst = src_gray * (mask[:, :].astype(src.dtype))
    return dst[box[1]:box[3],box[0]:box[2]]

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Code for Refined Bounding Boxes')
    parser.add_argument("--image_folder", type=str, default="images", help="path to dataset")
    parser.add_argument("--model_def", type=str, default="../PyTorch-YOLOv3/config/yolov3.cfg",
                        help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default="../PyTorch-YOLOv3/weights/yolov3.weights",
                        help="path to weights file")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
    args = parser.parse_args()


    # Set up model and device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Darknet(args.model_def, img_size=args.img_size).to(device)
    model.load_darknet_weights(args.weights_path)
    model.eval()

    # make a dataloader for the image folder
    dataloader = DataLoader(
            ImageFolder(args.image_folder, img_size=args.img_size),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.n_cpu,
        )
    # get the bounding boxes from yolo
    boxes = getBoundingBoxes(dataloader, args)

    # extract the edge masks from the parts of the image inside the bounding boxes
    masks=[]
    for batch_i, (img_paths, input_imgs) in enumerate(dataloader):
        src = cv2.imread(img_paths[0])
        src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        for box in boxes[batch_i]:
            # import pdb; pdb.set_trace()
            print(box)
            masks.append(CannyThreshold(box, src_gray, args))

    for m in masks:
        cv2.imshow('',m)
        cv2.waitKey()


