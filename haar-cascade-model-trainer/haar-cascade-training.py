import cv2
import numpy as np
import os

# Path to the directory containing positive samples
current_dir = os.path.dirname(os.path.abspath(__file__))
pos_dir = os.path.join(current_dir, 'positive_samples/cars')
# Path to the directory where positive samples will be created
pos_out_dir = os.path.join(current_dir, 'positive_samples_created/')

# Define the parameters for creating positive samples
num_samples = 100  # Number of positive samples to create for each image
width = 64  # Width of positive samples
height = 64  # Height of positive samples
min_scale = 0.8  # Minimum scale factor for creating positive samples
max_scale = 1.2  # Maximum scale factor for creating positive samples

# Loop over each image in the positive samples directory
for filename in os.listdir(pos_dir):
    if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png') or filename.endswith('.webp'):
        # Load the image
        img = cv2.imread(os.path.join(pos_dir, filename))

        # Loop over the number of positive samples to create for this image
        for i in range(num_samples):
            # Randomly generate a scale factor for the positive sample
            scale = np.random.uniform(min_scale, max_scale)

            # Randomly generate a rotation angle for the positive sample
            angle = np.random.uniform(-10, 10)

            # Generate a transformation matrix for the positive sample
            M = cv2.getRotationMatrix2D((width/2, height/2), angle, scale)

            # Apply the transformation to the positive sample
            sample = cv2.warpAffine(img, M, (width, height))

            # Save the positive sample to the output directory
            sample_filename = os.path.splitext(filename)[0] + '_sample{}.jpg'.format(i)
            cv2.imwrite(os.path.join(pos_out_dir, sample_filename), sample)
