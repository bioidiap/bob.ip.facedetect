.. _bob.ip.facedetect.tinyface:

==============================
 Face detection using TinyFace
==============================

This package comes with a TinyFace face detector. The Original Model is ``ResNet101`` 
from https://github.com/peiyunh/tiny. Please check for more details on TinyFace. The 
model is converted into MxNet Interface and the code used to implement the model are 
from https://github.com/chinakook/hr101_mxnet.


See below for an example on how to use
:any:`bob.ip.facedetect.tinyface.TinyFacesDetector`:

.. literalinclude:: plot/detect_faces_tinyface.py
   :linenos:

.. figure:: img/detect_faces_tinyface.png
  :figwidth: 75%
  :align: center
  :alt: Multi-Face Detection results using TinyFace.

  Multiple faces are detected by TinyFace.
