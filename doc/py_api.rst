.. vim: set fileencoding=utf-8 :
.. Manuel Guenther <manuel.guenther@idiap.ch>
.. Wed 14 Jan 16:15:27 CET 2015


============
 Python API
============

Classes
-------

.. autosummary::

   bob.ip.facedetect.BoundingBox
   bob.ip.facedetect.FeatureExtractor
   bob.ip.facedetect.Cascade
   bob.ip.facedetect.Sampler
   bob.ip.facedetect.TrainingSet

Functions
---------

.. autosummary::

   bob.ip.facedetect.detect_single_face
   bob.ip.facedetect.best_detection
   bob.ip.facedetect.overlapping_detections
   bob.ip.facedetect.prune_detections
   bob.ip.facedetect.expected_eye_positions


Constants
---------

.. py:data:: bob.ip.facedetect.default_cascade

   The pre-trained cascade file that is used by all scripts.


Detailed Information
--------------------

.. automodule:: bob.ip.facedetect
