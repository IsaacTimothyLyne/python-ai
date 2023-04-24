import os
import cv2
from random import randrange as rand

class FaceDetector:
    def __init__(self, xml_path):
        self.trained_face_data = cv2.CascadeClassifier(xml_path)

    def detect_faces(self, img):
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return self.trained_face_data.detectMultiScale(gray_img)

class FaceDrawer:
    def __init__(self, shape='rect'):
        self.shape = shape

    def draw(self, img, coordinates):
        for (x, y, w, h) in coordinates:
            if self.shape == 'rect':
                cv2.rectangle(img, (x, y), (x+w, y+h),
                              (rand(128,256), rand(128,256), rand(128,256)), 2)
            elif self.shape == 'circle':
                center = (x + w//2, y + h//2)
                radius = min(w, h) // 2
                cv2.circle(img, center, radius,
                           (rand(128,256), rand(128,256), rand(128,256)), 2)
            else:
                raise ValueError('The outline shape value is not defined')

class WebcamFaceDetector:
    def __init__(self, face_detector, face_drawer):
        self.face_detector = face_detector
        self.face_drawer = face_drawer
        self.webcam = cv2.VideoCapture(0)

    def __del__(self):
        cv2.destroyAllWindows()
        self.webcam.release()

    def detect_faces(self):
        success, frame = self.webcam.read()
        if not success:
            raise RuntimeError('Error getting a frame from the webcam, there may not be a webcam available')
        coordinates = self.face_detector.detect_faces(frame)
        self.face_drawer.draw(frame, coordinates)
        cv2.imshow('window name', frame)

    def run(self):
        while True:
            self.detect_faces()
            key_pressed = cv2.waitKey(1) # delay
            # quit with Q or q
            if key_pressed == 81 or key_pressed == 113:
                break

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    xml_path = os.path.join(current_dir, 'haarcascade_frontalface_default.xml')
    face_detector = FaceDetector(xml_path)
    face_drawer = FaceDrawer()
    webcam_face_detector = WebcamFaceDetector(face_detector, face_drawer)
    webcam_face_detector.run()

if __name__ == '__main__':
    main()
