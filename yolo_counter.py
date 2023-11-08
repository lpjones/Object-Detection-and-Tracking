import os
import cv2
from sort import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""
    1. Parse input files <YOLO weights file> <YOLO config file> <Label file> <Path to input video>
    
    2. Modify code so we only draw/annotate boxes for people

    3. Use MOTs (SORT tracker) to assign ID's to bound boxes, refer to sort.md and sort.py 

    4. Save new video with bounding boxes and counter for number of people to "mot_vid/MOTS20-09-result.mp4"
    
"""

# Parse in arguments from command line
# Referenced from argparse library
parser = argparse.ArgumentParser(description="Command-Line Parser")

# Define a positional argument, our input_file
parser.add_argument('input_files', nargs=4, type=str, help='<Weights file>   <config file>   <labels file>  <Video Path>')
parser.add_argument('-inf', action = "store_true", help="print average inference time")
parser.add_argument('-classes_all', action = "store_true", help="print all classes detected")
parser.add_argument('-classes_im', action = "store_true", help = "print classes per image")


args = parser.parse_args()


def check_input_paths(args):
    # Check for correct num of arguments
    if len(args.input_files) != 4:
        print("Invalid number of input files. Need 4\n \
                <Weights file>   <config file>   <labels file>  <Video Path>")
        exit(0)

    # Check for valid file paths
    if not os.path.isfile(args.input_files[3]):
        print(f"{args.input_files[3]} is not a valid file path")
        exit(0)

    video = args.input_files[3]

    try:
        labels = pd.read_csv(args.input_files[2], header=None)
    except:
        print(f"Invalid Labels file {args.input_files[2]}")
        exit(0)
    return video, labels

# Load YOLO model
def load_YOLO(model_architecture, model_weights):
    try:
        network = cv2.dnn.readNetFromDarknet(model_architecture, model_weights)
    except:
        print("Invalid Weights or Config file")
        exit(0)
    yolo_layers = ['yolo_82', 'yolo_94', 'yolo_106']
    return yolo_layers, network


# Return array of all frames in video
def read_video(video_path):
    frames = []
    video = cv2.VideoCapture(video_path)
    if video.isOpened() == False:
            print(f"Invalid Video {video_path}")
            exit(0)

    while(True):
        ret, frame = video.read()
        
        if not ret: # TODO: See how to implement an end of video
            break
        
        frames.append(frame) # Store frame into array
        
    return frames


# Convert Images to Blobs
def im2Blob(im):
    input_blob = cv2.dnn.blobFromImage(im, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    return [input_blob] # Add dimension so that output is (1, 3, 416, 416)


# Pass images through the network
def imgs_to_network(blob_imgs, yolo_layers, network):
    outputs = []
    times = []
    for im in blob_imgs:
        #print(np.array(im).shape)
        network.setInput(im)
        start_time = time.time()
        output = network.forward(yolo_layers)
        end_time = time.time()
        times.append(end_time - start_time)
        outputs.append(output)
    return outputs, sum(times) / len(times)

video, labels = check_input_paths(args)

yolo_layers, network = load_YOLO(args.input_files[1], args.input_files[0])

video_frames = read_video(video)

def get_bounding_boxes(image, output):
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

    return bounding_boxes, classes, confidences, probability_minimum, threshold

def draw_tracking_results(image, box_tracker, labels, classes, confidences, probability_minimum, threshold, max):

    # Draw bounding boxes and information on images
    results = cv2.dnn.NMSBoxes(bounding_boxes, confidences, probability_minimum, threshold) # NMSBoxes filters out duplicate boxes

    coco_labels = 80
    np.random.seed(42)
    colours = np.random.randint(0, 255, size=(coco_labels, 3), dtype='uint8')

    filtered_bound_boxes = [] # Only contains bounding boxes with people in them
    if len(results) > 0:
        for i in results.flatten(): 
            if labels[0][classes[i]] == 'person': 
                filtered_bound_boxes.append(bounding_boxes[i])

    # Convert format to put into trackers update function
    for filtered_box in filtered_bound_boxes:
        filtered_box[2] = filtered_box[0] + filtered_box[2]
        filtered_box[3] = filtered_box[1] + filtered_box[3]

    trackers = box_tracker.update(np.array(filtered_bound_boxes)) # Pass boxes into the tracker
    trackers = trackers[::-1] 

    for i in range(len(trackers)):

        x_min, y_min = int(trackers[i][2]), int(trackers[i][3])
        x_max, y_max = int(trackers[i][0]), int(trackers[i][1])
        colour_box = (0, 255, 255)

        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), colour_box, 5)

        text_box = '{}'.format(int(trackers[i][4])) # Print out ID of tracked person

        cv2.putText(image, text_box, (x_max, y_max - 7), cv2.FONT_HERSHEY_SIMPLEX, .75, colour_box, 2)

    if int(trackers[-1][4]) > max:
        max = int(trackers[-1][4])
        
    person_counter_text_box =  'People: {}'.format(max)

    cv2.putText(image, person_counter_text_box, (np.array(video_frames)[0].shape[1]//2-20, 20), cv2.FONT_HERSHEY_SIMPLEX, .75, (0,255,0), 2)
    return image, max

video_dimm = np.array(video_frames)[0].shape[0:2]

boxed_frame_arr = [] 
box_tracker = Sort() # Create instance of SORT
max = 0

video_frames = video_frames[0:15]
for i, frame in enumerate(video_frames): # Loop through all frames in the video

    imgs_blob = im2Blob(frame) # Convert to blob

    outputs, avg_time = imgs_to_network(imgs_blob, yolo_layers, network) # Inferences are the outputs

    bounding_boxes, classes, confidences, probability_minimum, threshold = get_bounding_boxes(frame, outputs[0])

    boxed_frame, max = draw_tracking_results(frame, box_tracker, labels, classes, confidences, probability_minimum, threshold, max)

    boxed_frame_arr.append(boxed_frame)

# Show the video, used boxed_frame_arr to make video
result = cv2.VideoWriter('mot_vid/MOTS20-09-result.mp4',  
                         cv2.VideoWriter_fourcc(*'mp4v'), 
                         30.0, video_dimm[::-1]) 

for frame in boxed_frame_arr:
    result.write(frame)
    
  
# When everything done, release  
# the video capture and video  
# write objects 
result.release() 