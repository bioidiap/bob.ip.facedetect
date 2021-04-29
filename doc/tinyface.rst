
.. _bob.ip.facedetect.tinyface:

============================
 Face detection using TinyFaceN
============================

This package comes with a TinyFace face detector. The Original Model is ``ResNet101`` 
from https://github.com/peiyunh/tiny. Please check for more details on TinyFace. The 
model is converted into MxNet Interface and the code used to implement the model are 
from https://github.com/chinakook/hr101_mxnet.


See below for an example on how to use
:any:`bob.ip.facedetect.mtcnn.MTCNN`:

.. plot:: plot/detect_faces_tinyface.py
   :include-source: True

