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
# img_path = os.path.join(current_dir, 'myface.jpg')
# img = cv2.imread(img_path)
webCam = cv2.VideoCapture(0)

def draw_outlines(coordinates, shape='rect'): 
        for (x, y, w, h) in coordinates:
            match shape:
                case 'rect':
                    cv2.rectangle(frame, (x, y), (x+w, y+h),
                                (0, 255, 0), 2)
                case 'circle':
                    center = (x + w//2, y + h//2)
                    radius = min(w, h) // 2
                    cv2.circle(frame, center, radius,
                            (0, 255, 0), 2)
                case _:
                    raise ValueError('The outline shape value is not defined')


#### Iterate forever over the frames captured by the webcam
while True:
    successful_frame_read, frame = webCam.read()
    if successful_frame_read != True:
        raise RuntimeError('error getting a frame from the webcam, there may not be a webcam available')

    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)
    
    draw_outlines(face_coordinates, 'circle')
    
    cv2.imshow('window name', frame)
    key_pressed = cv2.waitKey(1) # delay
    
    # quit with Q or q
    if key_pressed==81 or key_pressed==113:
        break
    
cv2.destroyAllWindows()
webCam.release()    