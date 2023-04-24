# Image classification only
import os
import cv2

class Tracker:
    def __init__(self, car_xml_path, pedestrian_xml_path):
        self.car_tracker = cv2.CascadeClassifier(car_xml_path)
        self.pedestrian_tracker = cv2.CascadeClassifier(pedestrian_xml_path)
        
    def detect_cars(self, img):
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return self.car_tracker.detectMultiScale(gray_img)
    def detect_pedestrians(self, img):
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return self.pedestrian_tracker.detectMultiScale(gray_img)
    
class RectBodyDrawer:
    def __init__(self, color=(255,0,0)):
        self.color = color
    
    def draw(self, img, coordinates):
        for (x,y,w,h) in coordinates:
            cv2.rectangle(img, (x,y), (x+w, y+h), self.color, 2)
            
class RectCarDrawer:
    def __init__(self, color=(0,255,0)):
        self.color = color
    
    def draw(self, img, coordinates):
        for (x,y,w,h) in coordinates:
            cv2.rectangle(img, (x,y), (x+w, y+h), self.color, 2)

class CameraDetector:
    def __init__(self, detector, car_drawer, body_drawer):
        self.car_drawer = car_drawer
        self.body_drawer = body_drawer
        self.detector = detector
        self.cam = cv2.VideoCapture(0)
        
    def __del__(self):
        cv2.destroyAllWindows()
        self.cam.release()
    
    def detect_objects(self):
        (read_successful, frame) = self.cam.read()
        if not read_successful:
            raise RuntimeError('Error getting current frame of webcam, there may be an invalid choice for webcam input')
        cars = self.detector.detect_cars(frame)
        pedestrians = self.detector.detect_pedestrians(frame)
        self.car_drawer.draw(frame, cars)
        self.body_drawer.draw(frame, pedestrians)
        cv2.imshow('Car-Pedestrian Tracker', frame)
    
    def run(self):
        while True:
            self.detect_objects()
            key_pressed = cv2.waitKey(1)
            if key_pressed == 81 or key_pressed == 113:
                break
            
def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    cars_xml = os.path.join(current_dir, 'haar_data/cars.xml')
    pedestrian_xml = os.path.join(current_dir, 'haar_data/haarcascade_fullbody.xml') 
    tracker = Tracker(cars_xml, pedestrian_xml)
    body_drawer = RectBodyDrawer()
    car_drawer = RectCarDrawer()
    cameraDetector = CameraDetector(tracker, car_drawer, body_drawer)
    cameraDetector.run()

if __name__ == '__main__':
    main()
        
