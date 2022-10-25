"""
Provides a real time inference with the video from webcam by default and depth map on the output
"""



import sys
sys.path.append('D:\\depthest')
import os
import numpy as np
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import cv2
from PIL import Image
import matplotlib.pyplot as plt
cmap = plt.cm.viridis
import time

from dataloaders import transforms
import argparse
cudnn.benchmark = True

global GPU
GPU = False

if GPU == True:
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'  # Set the GPU.
    device = 'cuda:0'
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Set the CPU
    device = 'cpu'


def load_depth_estimation_model(checkpoint_path):
    """
    This function load the trained depth estimation model
    Parameters
    ----------
    checkpoint_path: path to the trained depth estimation model

    Returns
    -------
    model: depth estimation model
    """
    checkpoint = torch.load(checkpoint_path, map_location=device) # load model checkpoint
    if type(checkpoint) is dict:
        model = checkpoint['model']
        print("=> loaded best depth estimation model (epoch {})".format(checkpoint['epoch']))
    else:
        model = checkpoint

    return model

def colored_depthmap(depth_pred, d_min=None, d_max=None):
    depth = np.squeeze(depth_pred.data.cpu().numpy())
    print("Predicted depth map ", (depth-1))
    if d_min is None:
        d_min = np.min(depth)
    if d_max is None:
        d_max = np.max(depth)
    depth_relative = (depth - d_min) / (d_max - d_min)
    color_depth= 255 * cmap(depth_relative)[:,:,:3] # HWC
    color_depth = Image.fromarray(color_depth.astype('uint8'))
    return depth, color_depth


def img_transform(rgb):
    """
    This function preprocess the image
    Parameters
    ----------
    rgb: RGB image

    Returns
    -------
    rgb_np: preprocessed image
    """
    transform = transforms.Compose([
       #transforms.CenterCrop((228, 304)),
       transforms.Resize((224, 224)),
    ])
    rgb_np = rgb
    rgb_np = np.asfarray(rgb_np, dtype='float') / 255 # normalization
    return rgb_np



def depth_prediction(model,bgr_img):
    """
    This function predict the depth map from BGR image
    Parameters
    ----------
    model: depth estimation model
    bgr_img: BGR image

    Returns
    -------
    color_depth: color encoded depth map
    """
    model.eval() # evaluation mode
    rgb=cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB) # RGB to BGR
    #rgb = np.transpose(rgb, (1, 2, 0))
    rgb=img_transform(rgb) # image preprocess
    to_tensor = transforms.ToTensor() # numpy to tensor
    input_tensor = to_tensor(rgb)
    while input_tensor.dim() < 3:
         input_tensor = input_tensor.unsqueeze(0)
    input_tensor = input_tensor.unsqueeze(0) # add additional dimension
    if GPU==True:
        input_tensor=input_tensor.cuda()
    with torch.no_grad():
        pred_depth_tensor = model(input_tensor) # predicted depth map tensor
    pred_depth,color_depth=colored_depthmap(pred_depth_tensor)
    return color_depth

def run_model(model,source):
    """
    This function run the inference
    Parameters
    ----------
    model: depth estimation model
    source: source of input such as image, video, webcam

    Returns
    -------
    None: nothing return. only for visualization
    """
    if source=="webcam":
        videocap = cv2.VideoCapture(0)
        success, image = videocap.read()
        img_counter = 0
        while success:
            success, image = videocap.read()
            start = time.time()
            color_depth = depth_prediction(model,image)
            end = time.time()
            totalTime = end - start
            fps = 1 / totalTime
            opencv_image = cv2.cvtColor(np.array(color_depth), cv2.COLOR_RGB2BGR)
            opencv_image = cv2.applyColorMap(opencv_image, cv2.COLORMAP_RAINBOW)
            #cv2.putText(image, f'inf_time: {int(totalTime)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
            #cv2.putText(opencv_image, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
            cv2.putText(image, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
            cv2.imshow("Predicted depth image",opencv_image)
            cv2.imshow('Image', image)
            cv2.waitKey(100)
            img_counter += 1
    else:
        print("Define source of input ( image / video / webcam )")






parser = argparse.ArgumentParser(description='Depth estimation inference')
#parser.add_argument('--model', metavar='DATA',help='path to the pretrained model')
parser.add_argument('--model', default='D:\\depthest\\results\\kitti.samples=0.modality=rgb.arch=MobileNetV3SkipAddS_NNConv5R.decoder=nnconv.criterion=l1.lr=0.01.bs=8.pretrained=True/model_best.pth.tar',
                      type=str, metavar='PATH',
                      help='path to the pretrained checkpoint (default: '')')
parser.add_argument('--source', default='webcam', metavar='DATA',help='type of source (webcam, image, video')

args = parser.parse_args()
print(args)

load_model = load_depth_estimation_model(args.model)
run_model(load_model,args.source)