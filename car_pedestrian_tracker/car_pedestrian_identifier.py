# Image classification only
import os
import cv2
current_dir = os.path.dirname(os.path.abspath(__file__))

# pre trained model by open cv for full body & car detection using haar cascade model
cars_xml = os.path.join(current_dir, 'haar_data/cars.xml')
car_tracker = cv2.CascadeClassifier(cars_xml)

pedestrian_xml = os.path.join(current_dir, 'haar_data/haarcascade_fullbody.xml') 
pedestrian_tracker = cv2.CascadeClassifier(pedestrian_xml)

# image i am using to test
img_file = os.path.join(current_dir, 'images/2.jpg') 

# create open cv image
img = cv2.imread(img_file)

# convert to black n white
img_grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cars = car_tracker.detectMultiScale(img_grayscale, 1.0048852, 10)
print(cars)
# draw rects around the cars
for (x,y,w,h) in cars:
    cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)

# show image and wait for key press
cv2.imshow('Window Name', img)
cv2.waitKey()

print('code completed')