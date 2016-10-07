import pkg_resources
import math

from .detector import Sampler, Cascade
from ._library import BoundingBox, prune_detections, group_detections, overlapping_detections

import bob.io.base
import numpy

def default_cascade():
  """Returns the :py:class:`bob.ip.facedetect.Cascade` that is loaded from the pre-trained cascade file provided by this package."""
  return Cascade(bob.io.base.HDF5File(pkg_resources.resource_filename("bob.ip.facedetect", "MCT_cascade.hdf5")))


def average_detections(detections, predictions, relative_prediction_threshold = 0.25):
  """average_detections(detections, predictions, [relative_prediction_threshold]) -> bounding_box, prediction

  Computes the weighted average of the given detections, where the weights are computed based on the prediction values.

  **Parameters:**

  ``detections`` : [:py:class:`BoundingBox`]
    The overlapping bounding boxes.

  ``predictions`` : [float]
    The predictions for the ``detections``.

  ``relative_prediction_threshold`` : float between 0 and 1
    Limits the bounding boxes to those that have a prediction value higher then ``relative_prediction_threshold * max(predictions)``

  **Returns:**

  ``bounding_box`` : :py:class:`BoundingBox`
    The bounding box which has been merged from the detections

  ``prediction`` : float
    The prediction value of the bounding box, which is a weighted sum of the predictions with minimum overlap
  """
  # remove the predictions that are too low
  prediction_threshold = relative_prediction_threshold * max(predictions)
  detections, predictions = zip(*[[d,p] for d,p in zip(detections, predictions) if p >= prediction_threshold])

  # turn remaining predictions into weights
  s = sum(predictions)
  weights = [p/s for p in predictions]
  # compute weighted average of bounding boxes
  top = sum(w * b.topleft_f[0] for w, b in zip(weights, detections))
  left = sum(w * b.topleft_f[1] for w, b in zip(weights, detections))
  bottom = sum(w * b.bottomright_f[0] for w, b in zip(weights, detections))
  right = sum(w * b.bottomright_f[1] for w, b in zip(weights, detections))

  # compute the average prediction value
  value = sum(w*p for w,p in zip(weights, predictions))

  # return the average bounding box
  return BoundingBox((top, left), (bottom-top, right-left)), value



def best_detection(detections, predictions, minimum_overlap = 0.2, relative_prediction_threshold = 0.25):
  """best_detection(detections, predictions, [minimum_overlap], [relative_prediction_threshold]) -> bounding_box, prediction

  Computes the best detection for the given detections and according predictions.

  This is achieved by computing a weighted sum of detections that overlap with the best detection (the one with the highest prediction), where the weights are based on the predictions.
  Only detections with according prediction values > 0 are considered.

  **Parameters:**

  ``detections`` : [:py:class:`BoundingBox`]
    The detected bounding boxes.

  ``predictions`` : [float]
    The predictions for the ``detections``.

  ``minimum_overlap`` : float between 0 and 1
    The minimum overlap (in terms of Jaccard :py:meth:`BoundingBox.similarity`) of bounding boxes with the best detection to be considered.

  ``relative_prediction_threshold`` : float between 0 and 1
    Limits the bounding boxes to those that have a prediction value higher then ``relative_prediction_threshold * max(predictions)``

  **Returns:**

  ``bounding_box`` : :py:class:`BoundingBox`
    The bounding box which has been merged from the detections

  ``prediction`` : float
    The prediction value of the bounding box, which is a weighted sum of the predictions with minimum overlap
  """
  # remove all negative predictions since they harm the calculation of the weights
  detections = [detections[i] for i in range(len(detections)) if predictions[i] > 0]
  predictions = [predictions[i] for i in range(len(predictions)) if predictions[i] > 0]

  if not detections:
    raise ValueError("No detections with a prediction value > 0 have been found")

  # keep only the bounding boxes with the highest overlap
  detections, predictions = overlapping_detections(detections, numpy.array(predictions), minimum_overlap)

  return average_detections(detections, predictions, relative_prediction_threshold)


def detect_single_face(image, cascade = None, sampler = None, minimum_overlap=0.2, relative_prediction_threshold = 0.25):
  """detect_single_face(image, [cascade], [sampler], [minimum_overlap], [relative_prediction_threshold]) -> bounding_box, quality

  Detects a single face in the given image, i.e., the one with the highest prediction value.

  **Parameters:**

  ``image`` : array_like (2D aka gray or 3D aka RGB)
    The image to detect a face in.

  ``cascade`` : str or :py:class:`Cascade` or ``None``
    If given, the cascade file name or the loaded cascade to be used.
    If not given, the :py:func:`default_cascade` is used.

  ``sampler`` : :py:class:`Sampler` or ``None``
    The sampler that defines the sampling of bounding boxes to search for the face.
    If not specified, a default Sampler is instantiated, which will perform a tight sampling.

  ``minimum_overlap`` : float between 0 and 1
    Computes the best detection using the given minimum overlap, see :py:func:`best_detection`

  ``relative_prediction_threshold`` : float between 0 and 1
    Limits the bounding boxes to those that have a prediction value higher then ``relative_prediction_threshold * max(predictions)``

  **Returns:**

  ``bounding_box`` : :py:class:`BoundingBox`
    The bounding box containing the detected face.

  ``quality`` : float
    The quality of the detected face, a value greater than 0.
  """

  if cascade is None:
    cascade = default_cascade()
  elif isinstance(cascade, str):
    cascade = Cascade(bob.io.base.HDF5File(cascade))

  if sampler is None:
    sampler = Sampler(patch_size = cascade.extractor.patch_size, distance=2, scale_factor=math.pow(2.,-1./16.), lowest_scale=0.125)

  if image.ndim == 3:
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
  bb, quality = best_detection(detections, predictions, minimum_overlap, relative_prediction_threshold)

  return bb, quality


def detect_all_faces(image, cascade = None, sampler = None, threshold = 0, overlaps = 1, minimum_overlap = 0.2, relative_prediction_threshold = 0.25):
  """detect_all_faces(image, [cascade], [sampler], [threshold], [overlaps], [minimum_overlap], [relative_prediction_threshold]) -> bounding_boxes, qualities

  Detects all faces in the given image, whose prediction values are higher than the given threshold.

  If the given ``minimum_overlap`` is lower than 1, overlapping bounding boxes are grouped, with the ``minimum_overlap`` being the minimum Jaccard similarity between two boxes to be considered to be overlapping.
  Afterwards, all groups which have less than ``overlaps`` elements are discarded (this measure is similar to the Viola-Jones face detector).
  Finally, :py:func:`average_detections` is used to compute the average bounding box for each of the groups, including averaging the detection value (which will, hence, usually decrease in value).

  **Parameters:**

  ``image`` : array_like (2D aka gray or 3D aka RGB)
    The image to detect a face in.

  ``cascade`` : str or :py:class:`Cascade` or ``None``
    If given, the cascade file name or the loaded cascade to be used to classify image patches.
    If not given, the :py:func:`default_cascade` is used.

  ``sampler`` : :py:class:`Sampler` or ``None``
    The sampler that defines the sampling of bounding boxes to search for the face.
    If not specified, a default Sampler is instantiated.

  ``threshold`` : float
    The threshold of the quality of detected faces.
    Detections with a quality lower than this value will not be considered.
    Higher thresholds will not detect all faces, while lower thresholds will generate false detections.

  ``overlaps`` : int
    The number of overlapping boxes that must exist for a bounding box to be considered.
    Higher values will remove a lot of false-positives, but might increase the chance of a face to be missed.
    The default value ``1`` will not limit the boxes.

  ``minimum_overlap`` : float between 0 and 1
    Groups detections based on the given minimum bounding box overlap, see :py:func:`group_detections`.

  ``relative_prediction_threshold`` : float between 0 and 1
    Limits the bounding boxes to those that have a prediction value higher then ``relative_prediction_threshold * max(predictions)``

  **Returns:**

  ``bounding_boxes`` : [:py:class:`BoundingBox`]
    The bounding box containing the detected face.

  ``qualities`` : [float]
    The qualities of the ``bounding_boxes``, values greater than ``threshold``.
  """
  if cascade is None:
    cascade = default_cascade()
  elif isinstance(cascade, str):
    cascade = Cascade(bob.io.base.HDF5File(cascade))

  if sampler is None:
    sampler = Sampler(patch_size = cascade.extractor.patch_size, distance=2, scale_factor=math.pow(2.,-1./16.), lowest_scale=0.125)

  if image.ndim == 3:
    image = bob.ip.color.rgb_to_gray(image)

  detections = []
  predictions = []
  # get the detection scores for the image
  for prediction, bounding_box in sampler.iterate_cascade(cascade, image, threshold):
    detections.append(bounding_box)
    predictions.append(prediction)

  if not detections:
    # No face detected
    return None

  # group overlapping detections
  if minimum_overlap < 1.:
    detections, predictions = group_detections(detections, predictions, minimum_overlap, threshold, overlaps)

    if not detections:
      return None

    # average them
    detections, predictions = zip(*[average_detections(b, q, relative_prediction_threshold) for b,q in zip(detections, predictions)])

  return detections, predictions
