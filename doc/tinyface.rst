
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

.. code-block:: python

    # Import and Call the class TinyFacesDetector
    from bob.ip.facedetect.tinyface import TinyFacesDetector
    detector = TinyFacesDetector()
    
    # Call the function detect to annotate the given image
    annotations = detector.detect(image)
    
    # The function will return two coordinates, topleft and bottomright, for each detected faces.
    topleft = annotations["topleft"]
    bottomright = annotations["bottomright"]
    
    # eyes locations are the estimated results, not the real one, so be careful to use.
    leye = annotations["leye"]
    reye = annotations["reye"]


This face detector can be used for detecting single or multiple faces. If there are more than one face, the first entry of the returned annotation supposed to be the largest face in the image. 
  
  
.. figure:: img/detect_faces_tinyface.png
  :figwidth: 75%
  :align: center
  :alt: Multi-Face Detection results using TinyFace.

  Multiple faces are detected by TinyFace.
