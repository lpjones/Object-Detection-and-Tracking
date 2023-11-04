import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import pandas as pd

"""
    1) Parse arguments passed to program (path to images, coco names (class) file,
       different types of outputs (classifications, how many of each classification))

    2) Load yolo model, convert image to blob

    3) Pass images into yolo model to make inferences

    4) Take output of yolo model and match classification number to corresponding name in coconames
        Draw the bounding boxes on the images using the regression output
        Save bounded box image with label names to output file in out_imgs
    
    Things to track:
        Total number of classes detected
        Average inference time
        How many of each class total
        How many of each class per image
"""

"""
    python3 yolo_img_detector.py  <Weights file>   <config file>   <labels file>  <Images/Image dir> \
                                -inf  -classes_im 

    Default:
        -inf
        -classes_all
    
    -h = This help menu
    
    -classes_im = Per Image Breakdown
        Per Image Breakdown
        sample.jpg => Car: 1
        
    -inf = Show average inference time
        Average Inference Time: 0.613 seconds

    -classes_all = show all classes detected
"""

# Parse in arguments from command line
# Referenced from argparse library
parser = argparse.ArgumentParser(description="Command-Line Parser")

# Define a positional argument, our input_file
parser.add_argument('input_files', nargs='+', type=str, help='<Weights file>   <config file>   <labels file>  <Images/Image dir>')
parser.add_argument('-inf', action = "store_true", help="print average inference time")
parser.add_argument('-classes_all', action = "store_true", help="print all classes detected")
parser.add_argument('-classes_im', action = "store_true", help = "print classes per image")


args = parser.parse_args()

# Check for correct num of arguments
if len(args.input_files) < 4:
    print("Invalid number of input files. Need at least 4\n \
            <Weights file>   <config file>   <labels file>  <Image/Images path>")
    exit(0)

imgs = []
# Check if input image path is a directory
if os.path.isdir(args.input_files[3]):
    # If it is a directory pull out images from directory
    for file in os.listdir(args.input_files[3]):
        if file.endswith('.jpg'):
            imgs.append(args.input_files[3] + file)
else:
    # Check for valid file paths
    for file in args.input_files:
        if not os.path.isfile(file):
            print(f"{file} is not a valid file path")
            exit(0)
    imgs = args.input_files[3:]

try:
    labels = pd.read_csv(args.input_files[2])
except:
    print(f"Invalid Labels file {args.input_files[2]}")
    exit(0)


if args.classes_im:
    print("classes_im is selected")
if args.classes_all:
    print("classes_all is selected")
if args.inf: # True value is stored in args.inf
    print("-inf is selected")

# Load YOLO model
def load_YOLO(model_architecture, model_weights):
    try:
        network = cv2.dnn.readNetFromDarknet(model_architecture, model_weights)
    except:
        print("Invalid Weights or Config file")
        exit(0)
    layers = network.getLayerNames()
    yolo_layers = ['yolo_82', 'yolo_94', 'yolo_106']


# Check if image file is valid
def read_img(image_path_arr):
    fig, ax = plt.subplots(1, len(image_path_arr), figsize=(20, 20), squeeze=False)
    
    for i, image_path in enumerate(image_path_arr):
        try:
            image = cv2.imread(image_path)
            ax[0][i].axis('off')
            ax[0][i].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        except:
            print(f"Invalid Input Image {image_path}")
            exit(0)
        
    plt.show()

load_YOLO(args.input_files[1], args.input_files[0])

read_img(imgs)

# Convert Image to Blob


# Pass image through the network


# Define Variables for drawing on image