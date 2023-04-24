# Face Detection
Face Detection from tutorial and corse at: https://www.youtube.com/watch?v=XIrOM9oP3pA

This script and program only works with front face images/videos, and has not been trained on side faces

## Information
All reference images given to the AI to learn by will be in black and white since the colour of skin or lighting doesnt change the face or any way that the face should be detected.

### Steps
- Get a crap load of faces
- Turn them all black and white/monochromatic
- Train the algorithm to detect each face as best as possible

### Breakdown
- Use OpenCv import
- Using a pre-trained model from https://github.com/opencv/opencv/tree/master/data/haarcascades
- 