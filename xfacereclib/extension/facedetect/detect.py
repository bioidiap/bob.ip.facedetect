import pkg_resources
import math

from .detector import Sampler, io
from ._features import prune_detections
from .utils import best_detection

default_cascade = pkg_resources.resource_filename("xfacereclib.extension.facedetect", "MCT_cascade.hdf5")

def detect_single_face(image, cascade = None, distance=2, scale_factor=math.pow(2.,-1./16.), lowest_scale=0.125, prediction_threshold=None, pruning_threshold=1, best_detection_overlap=0.2):

  sampler = Sampler(distance=distance, scale_factor=scale_factor, lowest_scale=lowest_scale)
  if cascade is None:
    cascade = default_cascade

  if isinstance(cascade, str):
    cascade = load(cascade)

  detections = []
  predictions = []
  # get the detection scores for the image
  for prediction, bounding_box in sampler.iterate_cascade(cascade, image):
    if prediction_threshold is None or prediction > prediction_threshold:
      detections.append(bounding_box)
      predictions.append(prediction)

  if not detections:
    return None

  # prune detections / or simply sort them
  detections, predictions = prune_detections(detections, predictions, pruning_threshold)

  # compute average over the best locations
  bb, value = best_detection(detections, predictions, best_detection_overlap)

  return bb

