#!/usr/bin/env python

import xfacereclib.extension.facedetect

cascade_file = xfacereclib.extension.facedetect.default_cascade

# define the preprocessor
preprocessor = xfacereclib.extension.facedetect.FaceDetector(
    cascade = cascade_file,
    post_processor = 'face-crop'
)
