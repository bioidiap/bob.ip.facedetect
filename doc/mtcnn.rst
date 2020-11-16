
.. _bob.ip.tensorflow_extractor.face_detect:

============================
 Face detection using MTCNN
============================

This package comes with a wrapper around the MTCNN (v1) face detector. See
https://kpzhang93.github.io/MTCNN_face_detection_alignment/index.html for more
information on MTCNN. The model is directly converted from the caffe model using code in
https://github.com/blaueck/tf-mtcnn

See below for an example on how to use
:any:`bob.ip.tensorflow_extractor.MTCNN`:

.. plot:: plot/detect_faces_mtcnn.py
   :include-source: True

