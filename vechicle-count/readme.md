## Importing Libraries
import cv2
import numpy as np

cv2: This is the OpenCV library, used for computer vision tasks.
numpy: A library for numerical operations in Python, used here for image processing tasks.

##  Video Capture Initialization
cap = cv2.VideoCapture('vechicle-count/video.mp4')

cv2.VideoCapture('vechicle-count/video.mp4'): This initializes the video capture from the specified video file. The cap object is used to read frames from this video.

## Define the Position of the Count Line
count_line_position = 550

count_line_position = 550: This sets the vertical position (in pixels) of a line on the video, which will be used to count the vehicles crossing it.


## Background Subtraction Algorithm Initialization
algo = cv2.bgsegm.createBackgroundSubtractorMOG()

cv2.bgsegm.createBackgroundSubtractorMOG(): This creates a background subtractor object using the MOG (Mixture of Gaussians) method. It will help in detecting moving objects by subtracting the background from the current frame.

## Define Minimum Rectangle Size for Vehicle Detection
min_width_react = 80
min_height_react = 80

min_width_react and min_height_react: These define the minimum width and height of a detected object (in pixels) to be considered as a vehicle. Anything smaller will be ignored.

## Helper Function to Get the Center of a Rectangle
def center_handle(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx, cy

center_handle(x, y, w, h): This function calculates the center of a rectangle (likely representing a detected vehicle). It returns the coordinates (cx, cy) of the center.

## Initializations
detect = []
offset = 6
counter = 0

detect = []: This list will store the centers of detected vehicles.
offset = 6: This is the allowable error (in pixels) when checking if a vehicle has crossed the counting line.
counter = 0: This will count the number of vehicles that have crossed the line.

## Main Loop for Processing Each Frame
while True:
    ret, frame1 = cap.read()

cap.read(): Reads the next frame from the video. ret is a boolean indicating success or failure in reading the frame, and frame1 is the actual image of the frame.

## Convert Frame to Grayscale and Blur
    grey = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey, (3, 3), 5)

cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY): Converts the frame to grayscale, which simplifies the processing.
cv2.GaussianBlur(grey, (3, 3), 5): Applies Gaussian blur to the grayscale frame to reduce noise and details, helping in smoother object detection.
## Apply Background Subtraction
    img_sub = algo.apply(blur)

algo.apply(blur): Applies the background subtraction algorithm on the blurred frame, resulting in an image where moving objects (vehicles) are highlighted.

## Morphological Transformations
    dilat = cv2.dilate(img_sub, np.ones((5, 5)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilatada = cv2.morphologyEx(dilat, cv2.MORPH_CLOSE, kernel)
    dilatada = cv2.morphologyEx(dilatada, cv2.MORPH_CLOSE, kernel)

cv2.dilate(img_sub, np.ones((5, 5))): Dilates the image, making the white regions (detected moving objects) more pronounced.

cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)): Creates a structuring element (kernel) in the shape of an ellipse, used for morphological operations.

cv2.morphologyEx(dilat, cv2.MORPH_CLOSE, kernel): Applies morphological closing (dilation followed by erosion) to remove small holes in the detected objects.

##  Find Contours
    counterShape, h = cv2.findContours(dilatada, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

cv2.findContours(dilatada, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE): Finds contours (boundaries) of the detected objects. counterShape holds these contours, which are later used to draw rectangles around detected vehicles

## Draw the Count Line on the Video
    cv2.line(frame1, (25, count_line_position), (1200, count_line_position), (255, 127, 0), 3)

cv2.line(frame1, (25, count_line_position), (1200, count_line_position), (255, 127, 0), 3): Draws a line across the video frame where vehicles will be counted when they cross it.

## Process Each Detected Object
    for (i, c) in enumerate(counterShape):
        (x, y, w, h) = cv2.boundingRect(c)
        validate_counter = (w >= min_width_react) and (h >= min_height_react)
        if not validate_counter:
            continue
        cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame1, "Vechicle" + str(counter), (x, y - 20), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 244, 0), 2)


        for (i, c) in enumerate(counterShape):: Iterates over all detected contours.

cv2.boundingRect(c): Calculates the bounding rectangle for each contour.
validate_counter: Checks if the detected object meets the minimum size criteria to be considered a vehicle.

cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2): Draws a rectangle around the detected vehicle.

cv2.putText(frame1, "Vechicle" + str(counter), (x, y - 20), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 244, 0), 2): Puts a label with the current vehicle counter above the rectangle.


## Find and Draw the Center of the Vehicle
        center = center_handle(x, y, w, h)
        detect.append(center)
        cv2.circle(frame1, center, 4, (0, 0, 255), -1)

center_handle(x, y, w, h): Calculates the center of the detected rectangle (vehicle).
detect.append(center): Adds the center point to the detect list.
cv2.circle(frame1, center, 4, (0, 0, 255), -1): Draws a small circle at the center of the detected vehicle.

## Count Vehicles Crossing the Line
        for (x, y) in detect:
            if y < (count_line_position + offset) and y > (count_line_position - offset):
                counter += 1
                cv2.line(frame1, (25, count_line_position), (1200, count_line_position), (0, 127, 255), 3)
                detect.remove((x, y))
                print("Vechicle counter:" + str(counter))


for (x, y) in detect:: Iterates over all detected centers.

if y < (count_line_position + offset) and y > (count_line_position - offset):: Checks if the vehicle center is within the counting line (considering some offset for error).

counter += 1: Increments the vehicle counter when a vehicle crosses the line.

cv2.line(frame1, (25, count_line_position), (1200, count_line_position), (0, 127, 255), 3): Changes the color of the line to indicate a vehicle has crossed.

detect.remove((x, y)): Removes the detected vehicle center from the list after counting.

 ## Display the Vehicle Counter on the Video
     cv2.putText(frame1, "Vechicle counter:" + str(counter), (450, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)

cv2.putText(frame1, "Vechicle counter:" + str(counter), (450, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5): Displays the total vehicle counter on the video frame.

## Display the Video
    cv2.imshow('Video Original', frame1)

cv2.imshow('Video Original', frame1): Displays the processed video frame in a window titled "Video Original".

## Exit on 'Enter' Key Press
    if cv2.waitKey(1) == 13:
        break

cv2.waitKey(1): Waits for 1 millisecond for a key press.
== 13: Checks if the 'Enter' key is pressed. If so, the loop breaks and the program exits.

## Release Resources and Close Windows
cv2.destroyAllWindows()
cap.release()

cv2.destroyAllWindows(): Closes all OpenCV windows.
cap.release(): Releases the video capture object.


------------------------ Happy Ending ---------------------
