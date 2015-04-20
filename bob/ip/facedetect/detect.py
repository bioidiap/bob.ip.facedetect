import pkg_resources
import math

from .detector import Sampler, Cascade
from ._library import BoundingBox, prune_detections, overlapping_detections

import bob.io.base
import numpy

def default_cascade():
  """Returns the :py:class:`Cascade` that is loaded from the pre-trained cascade file provided by this package."""
  return Cascade(bob.io.base.HDF5File(pkg_resources.resource_filename("bob.ip.facedetect", "MCT_cascade.hdf5")))


def best_detection(detections, predictions, minimum_overlap = 0.2):
  """best_detection(detections, predictions, [minimum_overlap]) -> bounding_box, prediction

  Computes the best detection for the given detections and according predictions.

  This is acheived by computing a weighted sum of detections that overlap with the best detection (the one with the highest prediction), where the weights are based on the predictions.
  Only detections with according prediction values > 0 are considered.

  **Keyword Parameters:**

  detections : [:py:class:`bob.ip.facedetect.BoundingBox`]
    The detected bounding boxes.

  predictions : [float]
    The predictions for the ``detections``.

  minimum_overlap : float between 0 and 1
    The minimum overlap of bounding boxes with the best detection to be considered.

  **Returns:**

  bounding_box : :py:class:`bob.ip.facedetect.BoundingBox`
    The bounding box which has been merged from the detections

  prediction : float
    The prediction value of the bounding box, which is a weighted sum of the predictions with minimum overlap
  """
  # remove all negative predictions since they harm the calculation of the weights
  detections = [detections[i] for i in range(len(detections)) if predictions[i] > 0]
  predictions = [predictions[i] for i in range(len(predictions)) if predictions[i] > 0]

  if not detections:
    raise ValueError("No detections with a prediction value > 0 have been found")

  # keep only the bounding boxes with the highest overlap
  detections, predictions = overlapping_detections(detections, numpy.array(predictions), minimum_overlap)

  # compute the mean of the detected bounding boxes
  s = sum(predictions)
  weights = [p/s for p in predictions]
  top = sum(w * b.topleft_f[0] for w, b in zip(weights, detections))
  left = sum(w * b.topleft_f[1] for w, b in zip(weights, detections))
  bottom = sum(w * b.bottomright_f[0] for w, b in zip(weights, detections))
  right = sum(w * b.bottomright_f[1] for w, b in zip(weights, detections))

  value = sum(w*p for w,p in zip(weights, predictions))
  # as the detection value, we use the *BEST* value of all detections.
#  value = predictions[0]

  return BoundingBox((top, left), (bottom-top, right-left)), value


def detect_single_face(image, cascade = None, sampler = None, minimum_overlap=0.2):
  """detect_single_face(image, [cascade], [sampler], [minimum_overlap]) -> bounding_box, quality

  Detects a single face in the given image, i.e., the one with the highest prediction value.

  **Keyword Parameters:**

  image : array_like (2D aka gray or 3D aka RGB)
    The image to detect a face in.

  cascade : str or :py:class:`bob.ip.facedetect.Cascade` or ``None``
    If given, the cascade file name or the loaded cascase to be used.
    If not given, the :py:func:`bob.ip.facedetect.default_cascade` is used.

  sampler : :py:class:`bob.ip.facedetect.Sampler` or ``None``
    The sampler that defines the sampling of bounding boxes to search for the face.
    If not specified, a default Sampler is instantiated, which will perform a tight sampling.

  minimum_overlap : float between 0 and 1
    Computes the best detection using the given minimum overlap, see :py:func:`bob.ip.facedetect.best_detection`

  **Return value:**

  bounding_box : :py:class:`bob.ip.facedetect.BoundingBox`
    The bounding box containing the detected face.

  quality : float
    The quality of the detected face, a value greater than 0.
  """

  if cascade is None:
    cascade = default_cascade()
  elif isinstance(cascade, str):
    cascade = Cascade(bob.io.base.HDF5File(cascade))

  if sampler is None:
    sampler = Sampler(patch_size = cascade.extractor.patch_size, distance=2, scale_factor=math.pow(2.,-1./16.), lowest_scale=0.125)

  if len(image.shape)==3:
    image = bob.ip.color.rgb_to_gray(image)

  detections = []
  predictions = []
  # get the detection scores for the image
  for prediction, bounding_box in sampler.iterate_cascade(cascade, image, None):
    detections.append(bounding_box)
    predictions.append(prediction)

  if not detections:
    return None

  # compute average over the best locations
  bb, quality = best_detection(detections, predictions, minimum_overlap)

  return bb, quality


def detect_all_faces(image, cascade = None, sampler = None, threshold = 0, minimum_overlap=0.2):
  """detect_all_faces(image, [cascade], [sampler], [minimum_overlap]) -> bounding_boxes, qualities

  Detects a single face in the given image, i.e., the one with the highest prediction value.

  **Keyword Parameters:**

  image : array_like (2D aka gray or 3D aka RGB)
    The image to detect a face in.

  cascade : str or :py:class:`bob.ip.facedetect.Cascade` or ``None``
    If given, the cascade file name or the loaded cascase to be used.
    If not given, the :py:func:`bob.ip.facedetect.default_cascade` is used.

  sampler : :py:class:`bob.ip.facedetect.Sampler` or ``None``
    The sampler that defines the sampling of bounding boxes to search for the face.
    If not specified, a default Sampler is instantiated.

  threshold : float
    The threshold of the quality of detected faces.
    Detections with a quality lower than this value will not be considered.
    Higher thresholds will not detect all faces, while lower thresholds will generate false detections.

  minimum_overlap : float between 0 and 1
    Computes the best detection using the given minimum overlap, see :py:func:`bob.ip.facedetect.best_detection`

  **Return value:**

  bounding_boxes : [:py:class:`bob.ip.facedetect.BoundingBox`]
    The bounding box containing the detected face.

  quality : [float]
    The quality of the detected face, a value greater than 0.
  """
  if cascade is None:
    cascade = default_cascade()
  elif isinstance(cascade, str):
    cascade = Cascade(bob.io.base.HDF5File(cascade))

  if sampler is None:
    sampler = Sampler(patch_size = cascade.extractor.patch_size, distance=2, scale_factor=math.pow(2.,-1./16.), lowest_scale=0.125)

  if len(image.shape)==3:
    image = bob.ip.color.rgb_to_gray(image)

  detections = []
  predictions = []
  # get the detection scores for the image
  for prediction, bounding_box in sampler.iterate_cascade(cascade, image, threshold):
    detections.append(bounding_box)
    predictions.append(prediction)

  if not detections:
    return None

  # prune overlapping detections
  bbs, qualities = prune_detections(detections, predictions, minimum_overlap)

  return bbs, qualities
