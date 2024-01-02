import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import pandas as pd
import time

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



def check_input_paths(args):
    imgs = []

    # Check for correct num of arguments
    if len(args.input_files) < 4:
        print("Invalid number of input files. Need at least 4\n \
                <Weights file>   <config file>   <labels file>  <Image/Images path>")
        exit(0)

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
        labels = pd.read_csv(args.input_files[2], header=None)
    except:
        print(f"Invalid Labels file {args.input_files[2]}")
        exit(0)
    return imgs, labels

# Load YOLO model
def load_YOLO(model_architecture, model_weights):
    try:
        network = cv2.dnn.readNetFromDarknet(model_architecture, model_weights)
    except:
        print("Invalid Weights or Config file")
        exit(0)
    yolo_layers = ['yolo_82', 'yolo_94', 'yolo_106']
    return yolo_layers, network


# Check if image file is valid
def read_img(image_path_arr):
    fig, ax = plt.subplots(1, len(image_path_arr), figsize=(20, 20), squeeze=False)
    images_cv2 = []
    for i, image_path in enumerate(image_path_arr):
        try:
            image = cv2.imread(image_path)
            images_cv2.append(image)
            ax[0][i].axis('off')
            ax[0][i].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        except:
            print(f"Invalid Input Image {image_path}")
            exit(0)
    
    #plt.show()
    return images_cv2



# Convert Images to Blobs
def im2Blob(images_cv2):
    imgs_blob = []
    for im in images_cv2:
        input_blob = cv2.dnn.blobFromImage(im, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        imgs_blob.append(input_blob)
    return imgs_blob


# Pass images through the network
def imgs_to_network(blob_imgs, yolo_layers, network):
    outputs = []
    times = []
    for im in blob_imgs:
        network.setInput(im)
        start_time = time.time()
        output = network.forward(yolo_layers)
        end_time = time.time()
        times.append(end_time - start_time)
        outputs.append(output)
    return outputs, sum(times) / len(times)



def draw_image(image, output, labels):
    # Define variables for drawing on image
    bounding_boxes = []
    confidences = []
    classes = []
    probability_minimum = 0.5
    threshold = 0.3
    h, w = image.shape[:2]
    # Get bounding boxes, confidences and classes
    for result in output:
        for detection in result:
            scores = detection[5:]
            class_current = np.argmax(scores)
            confidence_current = scores[class_current]
            if confidence_current > probability_minimum:
                box_current = detection[0:4] * np.array([w, h, w, h])
                x_center, y_center, box_width, box_height = box_current.astype('int')
                x_min = int(x_center - (box_width / 2))
                y_min = int(y_center - (box_height / 2))
                bounding_boxes.append([x_min, y_min, int(box_width), int(box_height)])
                confidences.append(float(confidence_current))
                classes.append(class_current)

    # Draw bounding boxes and information on images
    results = cv2.dnn.NMSBoxes(bounding_boxes, confidences, probability_minimum, threshold)
    coco_labels = 80
    np.random.seed(42)
    colours = np.random.randint(0, 255, size=(coco_labels, 3), dtype='uint8')
    # Track number of each class predicted per image
    classes_dict = {}
    if len(results) > 0:
        for i in results.flatten(): # i is the value, , not the index of the results array
            x_min, y_min = bounding_boxes[i][0], bounding_boxes[i][1]
            box_width, box_height = bounding_boxes[i][2], bounding_boxes[i][3]
            colour_box = [int(j) for j in colours[classes[i]]]
            cv2.rectangle(image, (x_min, y_min), (x_min + box_width, y_min +    box_height),
                        colour_box, 5)
            text_box = labels[0][classes[i]] + ': {:.2f}'.format(confidences[i])
            cv2.putText(image, text_box, (x_min, y_min - 7), cv2.FONT_HERSHEY_SIMPLEX, .75, colour_box, 2)
            if labels[0][classes[i]] not in classes_dict:
                classes_dict[labels[0][classes[i]]] = 1 # If "person" not in dictionary, add it to dict, it starts w/ one instance
            else:
                classes_dict[labels[0][classes[i]]] += 1 # Get the current num of instances of "person," and increase it by 1, if you see in array [1, 2, 4, 1] in results, there are two persons in the  image
    return image, classes_dict


imgs, labels = check_input_paths(args)

yolo_layers, network = load_YOLO(args.input_files[1], args.input_files[0])

imgs_cv2 = read_img(imgs)

imgs_blob = im2Blob(imgs_cv2)

outputs, avg_time = imgs_to_network(imgs_blob, yolo_layers, network) # Inferences are the outputs

output_path = './out_imgs/'

# Create out_images/ if it doesn't exist
try:
    os.mkdir(output_path)
except:
    pass
dicts, img_names = [], []
classes_all_dict = {} # Shows total class instances across all images
# save images with boxes as files in out_images/
for i, im in enumerate(imgs_cv2):
    img, class_dict = draw_image(im, outputs[i], labels) # Draw inferences made in imgs_to_network() to the original images (imgs_cv2)
    dicts.append(class_dict)
    # Extract file name with extension
    file = str(os.path.basename(imgs[i]))
    img_names.append(file)
    # Remove extension
    file = str(os.path.splitext(file)[0])
    # Save to file
    plt.imsave(output_path + file + '_out.jpg', cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

if args.classes_im: # If flag -classes_im is true
    print('Per Image Breakdown')
    for i, im in enumerate(dicts):
        print(img_names[i]) # Print current image name
        for cl in im:
            print(cl + ':', im[cl]) # Print out class and value of that class found in current image
        print('\n', end='')
if args.classes_all: # If flag -classes_all is true
    print('Total Detection Breakdown') # Across all the images you scanned, how many of each class did you see in total?
    # Go through dict of each image and add to total
    for im in dicts: # dicts is array of dicts, each image has its own dict
        for cl in im: # For classes encountered in current image dict
            if cl not in classes_all_dict: 
                classes_all_dict[cl] = im[cl] # For the curr image you are on, add the curr class you are on (that has not yet been encountered)
            else:
                classes_all_dict[cl] += im[cl] # For the curr i
    # Print out classes with how many of each
    for cl in classes_all_dict:
        print(cl + ':', classes_all_dict[cl])
    print('\nTotal Number of Classes Detected:', len(classes_all_dict))
if args.inf: # True value is stored in args.inf
    print(f"Average Inference Time: %.3f"% avg_time)