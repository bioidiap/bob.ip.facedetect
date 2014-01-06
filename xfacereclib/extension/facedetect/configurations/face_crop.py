#!/usr/bin/env python

import xfacereclib.extension.facedetect
import pkg_resources

# Cropping
CROPPED_IMAGE_HEIGHT = 80
CROPPED_IMAGE_WIDTH  = CROPPED_IMAGE_HEIGHT * 4 / 5

cascade_file = pkg_resources.resource_filename("xfacereclib.extension.facedetect", 'MCT_cascade.hdf5')

# define the preprocessor
preprocessor = xfacereclib.extension.facedetect.FaceDetector(
    cropped_image_size = (CROPPED_IMAGE_HEIGHT, CROPPED_IMAGE_WIDTH),
    cascade = cascade_file,
    post_processor = 'face-crop'
)
