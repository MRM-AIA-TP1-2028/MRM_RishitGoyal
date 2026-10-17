Blue Object Selection and Tennis Ball Detection using OpenCV

1. Approach to Deal with the Problem:
The problem was approached by developing a real-time object detection and tracking system using Python
and OpenCV. The code detects blue objects of any shape and a tennis ball separately from a live camera
feed. It calculates their area, bounding box, center coordinates, and radius , while tracking blue-object
movement and displaying its relative position to the tennis ball.

2. Method Used to Solve the Problem:
The method uses HSV color thresholding to separate blue objects and the yellow-green tennis ball from the
background. Contour detection identifies objects and calculates their geometric properties. Circularity is
used to help identify the tennis ball, while centroid-based matching tracks blue objects across frames.
OpenCV displays the results, including object measurements, movement arrows, and the distance between object centers.

3. Other Possible Methods:
Alternative methods include RGB thresholding, background subtraction, shape-based detection, template
matching, deep-learning models such as YOLO, and optical flow. For improved tracking, Kalman filters and
Hungarian assignment can be used. The current HSV and contour-based method is simple and
computationally efficient, while deep learning and advanced tracking methods may perform better in
complex lighting conditions or crowded scenes.
