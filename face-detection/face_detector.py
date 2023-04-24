import os
import cv2  # open source computer vision library
from random import randrange as rand  # just for colors for fun
# open cv has alot of libraries already setup for computer vision and already trained models

# list of pre-trained haar cascades (https://github.com/opencv/opencv/tree/master/data/haarcascades)

# get the path of the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# open cv pre trained data for frontal face clarification and detection
# (haar cascade algorithm)
xml_path = os.path.join(current_dir, 'haarcascade_frontalface_default.xml')
trained_face_data = cv2.CascadeClassifier(xml_path)

# path to the image file to detect faces in
img_path = os.path.join(current_dir, 'myface.jpg')
img = cv2.imread(img_path)

# change the image from imread to grayscale (for some reason instead of rgb its bgr)
grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# detect faces in image
face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

# for own learning purposes:
# gives an array like so: [[312 232 182 182]] will need to split?
# after research it is an array and has commas however as a numpy array no commas are displayed
print("Face Coordinates: ", face_coordinates)
# the first two values are the x and y coordinates of the top-left corner,
# and the last two values are the width and height of the rectangle.
# and to make the width we need to add the width to the x value
# so it makes the correct width according to the x value

# draw the rectangle('s) using the co-ordinates [[coordinates1],[coordinates2]]
# image, two points, color, thickness


def draw_outlines(coordinates, shape='rect'):  # my OOP implementation
    for (x, y, w, h) in coordinates:
        match shape:
            case 'rect':
                cv2.rectangle(img, (x, y), (x+w, y+h),
                              (rand(128,256), rand(128,256), rand(128,256)), 2)
            case 'circle':
                center = (x + w//2, y + h//2)
                radius = min(w, h) // 2
                cv2.circle(img, center, radius,
                           (rand(128,256), rand(128,256), rand(128,256)), 2)
            case _:
                raise ValueError('The outline shape value is not defined')


draw_outlines(face_coordinates)

# show the image
cv2.imshow('Window Name', img)

# pause execution so it can wait and show the image until a key is pressed
cv2.waitKey()

# clean up
cv2.destroyAllWindows()

print('code completed')
