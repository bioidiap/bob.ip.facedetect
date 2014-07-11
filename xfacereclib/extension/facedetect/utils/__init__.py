from boundingbox import bounding_box_from_annotation, expected_eye_positions, best_detection
from database import training_image_annot, test_image_annot
from .._features import BoundingBox

import numpy
import facereclib
import bob

def sqr(x):
  """This function computes the square of the given value.
  (I don't know why such a function is not part of the math module)."""
  return x*x

def irnd(x):
  """This function returns the integer value of the rounded given float value."""
  return int(round(x))


def detect_landmarks(localizer, image, bounding_box):
  scales = [1., 0.9, 0.8, 1.1, 1.2]
  shifts = [0, 0.1, 0.2, -0.1, -0.2]

  uint8_image = image.astype(numpy.uint8)
  # make the bounding box square shape by extending the horizontal position by 2 pixels times width/20
  corrected_bounding_box = BoundingBox(top = bounding_box.top, left = bounding_box.left - bounding_box.width / 10., height = bounding_box.height, width = bounding_box.height)

  for scale in scales:
    bs = corrected_bounding_box.scale_centered(scale)
    for y in shifts:
      by = bs.shift(y * bs.height, 0)
      for x in shifts:
        bb = by.shift(0, x * bs.width)

        top = max(bb.top, 0)
        left = int(max(bb.left, 0))
        bottom = min(bb.bottom, image.shape[0]-1)
        right = int(min(bb.right, image.shape[1]-1))
        landmarks = localizer.localize(uint8_image, top, left, bottom-top+1, right-left+1)

        if len(landmarks):
          facereclib.utils.debug("Found landmarks with scale %1.1f, and shift %1.1fx%1.1f" % (scale, y, x))
          return landmarks

  return []


def localize(localizer, feature_extractor, image, bounding_box):
  feature_vector = numpy.ndarray(feature_extractor.number_of_features, numpy.uint16)

  # scale image such that bounding box is in the correct size
  scale = float(feature_extractor.patch_size[0]) / float(bounding_box.height_f)
  assert abs(scale - float(feature_extractor.patch_size[1]) / float(bounding_box.width_f)) < 1e-10
  scaled_bb = bounding_box.scale(scale)

  # extract features for given image patch
  feature_extractor.prepare(image, scale)
  scaled_image_shape = feature_extractor.image.shape

  # compute shifted versions of the bb and get the MEDIAN feature positions
#  shifts = range(-3,4)
  shifts = [0]
  predictions = numpy.ndarray((len(shifts)**2, localizer.outputs))

  i = 0
  for y in shifts:
    for x in shifts:
      shifted_bb = scaled_bb.shift(y,x)
      if not shifted_bb.is_valid_for(scaled_image_shape):
        continue

      feature_extractor.extract_single(shifted_bb, feature_vector)
      # compute the predicted location
      prediction = numpy.ndarray(localizer.outputs)
      localizer(feature_vector, predictions[i])
      i += 1

  prediction = numpy.median(predictions[:i], 0)

  # compute landmarks in image coordinates
  landmarks = []
  scale = 1./scale
  for i in range(0, localizer.outputs, 2):
    y = (prediction[i] * scale) + bounding_box.center[0]
    x = (prediction[i+1] * scale) + bounding_box.center[1]
    landmarks.append((y,x))

  return landmarks

def predict(graphs, image, bounding_box):
  # scale image such that bounding box is in the correct size
  scale = float(graphs.patch_size[0]) / float(bounding_box.height_f)
  assert abs(scale - float(graphs.patch_size[1]) / float(bounding_box.width_f)) < 1e-10
  scaled_bb = bounding_box.scale(scale)

  # extract features for given image patch
  scaled_image = bob.ip.scale(image, scale)

  # compute shifted versions of the bb and get the MEDIAN feature positions
#  shifts = range(-3,4)
  shifts = [0]
  predictions = numpy.ndarray((len(shifts)**2, graphs.number_of_predictions()))

  i = 0
  for y in shifts:
    for x in shifts:
      shifted_bb = scaled_bb.shift(y,x)
      if not shifted_bb.is_valid_for(scaled_image.shape):
        continue

      # compute the predicted location
      predictions[i] = graphs.predict(scaled_image, shifted_bb)
      i += 1

  prediction = numpy.median(predictions[:i], 0)

  # compute landmarks in image coordinates
  landmarks = []
  scale = 1./scale
  for i in range(0, graphs.number_of_predictions(), 2):
    y = (prediction[i] * scale) + bounding_box.top
    x = (prediction[i+1] * scale) + bounding_box.left
    landmarks.append((y,x))

  return landmarks

