Completed by ryuliou and lpjones
# EEC174 AY Project A2: Path Tracking & Object Removal
# How to use
Download yolov3.weights file at https://pjreddie.com/media/files/yolov3.weights and move the file to the yolo_files/ folder

Executable files: \
yolo_counter.py \
yolo_img_detector.py

yolo_counter.py: \
Takes in weights file, config file, labels  file, and video path

Takes in a video and counts how many people show up in the image

Usage:
```console
python3 yolo_counter.py yolo_files/yolov3.weights yolo_files/yolov3.cfg yolo_files/coco.names mot_vid/MOTS20-09-raw.mp4
```
Libraries: \
os \
opencv2 \
numpy \
pandas \
matplotlib

# Instructions (Do not clone this repo!)

Due Date: December 4, 2023

In this project, there are two seperate assignments. In the first, Path Tracking assignment you will implement a naive version of a tracker (similar to SORT) for assigning IDs and tracking path of objects in a video. In the second assignment, Object Removal, you will learn to use naive inpaint method to removal objects detected by yolo.

## Environment Setup

You are not required to use the server or docker for this lab. The requirements for this lab are same as Lab A1. Feel free to use same environment. Otherwise the requirements are:
- Python 3.x
- Numpy
- OpenCV (python)
- Matplotlib (python)

## Assignment 1: Path Tracking

Related files: ```tracker.py```, ```yolo_path_tracker.py```

This assignment is very closely related to Mini-Project A1, where instead of using SORT, you will develop your own basic tracker.
You are provided some skeleton code to read videos in ```yolo_path_tracker.py```. Please follow a similar structure in A1, where you must:

- Convert frame to blob
- Use YOLO net to detect objects
- Apply Non-Maximum Suppression (NMS) to filter out overlapping boxes
- Update the tracker with the centroids of detected objects
- Draw bounding boxes, trails, and labels on the frame

### Tracker
Up till tracker, your code can be identical to A1. In this assignment instead of suing sort.py, we will make out own tracker ```Tracker()``` class in ```tracker.py```.

You are provided with the class and method stubs and what each method should do. There is a breakdown of ```Tracker()```:

```Tracker()```
- Initialization (__init__):
    - Initialize variables to store object IDs, their centroids, lost count, and paths.
- Registering Objects (register):
    - Assign a unique ID to the new object.
    - Store its centroid and initialize its path and lost count.
- Deregistering Objects (deregister):
    - Remove an object's data from the tracker.
- Updating Objects (update):
    - For each new set of centroids (representing objects in the current frame):
        - If there are no tracked objects, register all new centroids.
        - If there are existing tracked objects, match new centroids to existing objects based on proximity.
        - Register any new objects that don't match existing ones.
        - Update the paths and lost count for each tracked object.
        - Deregister objects that have been lost for more than maxLost frames.
- Calculating Distance (_distance):
    - Implement the method to calculate the Euclidean distance between each pair of centroids from two lists.

We will match detected objects by their closest distance to last known. You can set a maximum distance radius to look for the object, otherwise deregister the object. 
The TA has set this to ```MAX_DISTANCE = 50```. So any distances beyond that will not be counted as same ID.

### Visualization

Please store the paths in tracker for IDs. You must visualize the paths by tracking the centroids of people detected. Implement a function that does this. Make sure to remove them beyond a certain length that you find is visually suitable. You can use ```cv.line()``` to plot the path. The path should be a list of centroids returned/stored in Tracker object.

Here is the TA's sample output video (or see kartik_sample/output.mp4:

![kartik_sample/kartik_out.gif](kartik_sample/kartik_out.gif)

## Assignment 2: Object Removal

In this part of the assignment, we will like A1, do yolo object detection on images. Instead of showing images with the detected objects, we will use the bounding boxes to find objects and remove them given surrouding pixel data. Please complete ```yolo_remover.py```

The program must be able to take 3 inputs:
1. The input image to remove objects from
2. The name of the class of ibkect to remove (you can get class names from class.names)
3. The pixel block size to sue to fill detected objects.

Your program will output similarly to A1, the imaage name with `_out.jpg`. The output image should have select class objects removed.

For detection please feel free to re-use codes from A1.

### Filling objects

You must write ```inpaint_yolo_box()```, which attempts to inpaint given bounding boxes in an image. 

1. Removal Logic:
    - Iterate over each bounding box.
    - For each row within a box, copy block of pixels from right to left.
    - Ensure the copied block does not exceed the left boundary of the bounding box.
    - Handle edge cases, especially when the remaining width is less than block_size.
2. Handling Image Edges:
    - Ensure your implementation does not attempt to access pixels outside the image boundaries.
3. Optimization:
    - Aim for an efficient implementation that can handle multiple bounding boxes in a single image.
4. Testing Your Function:
    - Test the function with images having various object sizes and locations.
    - You can use pre-defined bounding boxes for testing purposes.

Here is an example from TA ouput on beach.jpg:

![kartik_sample/beach_out.jpg](kartik_sample/beach_out.jpg)

## Evaluation

### Grade Breakdown

- Program Correctness: 80%
- Program usability, readability & design: 10%
- Interactive Grading: 10%

You are allowed to work in pairs for this assignment. Please avoid excessive collaboratation with other groups.

#### Program Corectness (80%)
Depending on implementation, your program may not match the TA outputs exactly. However, your functionality must be correct and achieve the same goal as the TA's outputs. There will be a lot of match in ouputs, but due to different implementation logic, do not expect exact outputs.

#### Program Usability, Readability & Design (10%)
Please make sure your programs are well commented and readable. Make sure to adress invalid CLI inputs or missing inputs. If the programs are used incorrectly, your programs should exit safetly and output the correct use. Have functions wherever necessary for readability. 

#### Interactive Grading (10%)
Since this is a group assignment, there will be interactive grading for this project. This is just to verify that all members understand the code fully. Hence, this will be individual -- to make sure you succeed, make sure you understand all of your submitted code, even if there are parts you did not code yourself. Interactive grading will be done during lab hours the week after assignment is due.

## Submission

Graded files

1. Path Tracking:
- ```tracker.py```
- ```yolo_path_tracker.py```

2. Object Removal:
- ```yolo_remover.py```

## Credits

Kartik Patwari, Jeff Lai, and Chen-Nee Chuah