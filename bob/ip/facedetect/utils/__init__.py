from boundingbox import bounding_box_from_annotation, expected_eye_positions, best_detection
from database import training_image_annot, test_image_annot
from .._features import BoundingBox

import numpy
import facereclib

import bob.ip.color
import bob.ip.draw
import bob.ip.base
import bob.core

def sqr(x):
  """This function computes the square of the given value.
  (I don't know why such a function is not part of the math module)."""
  return x*x

def irnd(x):
  """This function returns the integer value of the rounded given float value."""
  return int(round(x))



def display(image, annotations=None, color=(255,0,0), radius=5, clear=True):
  from matplotlib import pyplot
  if clear:
    pyplot.clf()
  if annotations is None:
    pyplot.imshow(image, cmap='gray')
  else:
    if len(image.shape) == 2:
      colored = bob.ip.color.gray_to_rgb(bob.core.convert(image, numpy.uint8, dest_range=(0,255), source_range=(min(0, numpy.min(image)), max(255, numpy.max(image)))))
    else:
      colored = image.copy()

    if isinstance(annotations, dict):
      annotations = [a for a in annotations.itervalues()]
    else:
      annotations = [(annotations[i], annotations[i+1]) for i in range(0, len(annotations), 2)]
    for a in annotations:
      try:
        bob.ip.draw.cross(colored, (int(a[0]), int(a[1])), radius=radius, color=color)
      except Exception as e:
        facereclib.utils.warn("Could not plot annotation %s" % str(a))
    pyplot.imshow(numpy.rollaxis(numpy.rollaxis(colored.astype(numpy.uint8), 2),2))
  pyplot.draw()
  return colored


def detect_landmarks(localizer, image, bounding_box):
  scales = [1., 0.9, 0.8, 1.1, 1.2]
  shifts = [0, 0.1, 0.2, -0.1, -0.2]

  uint8_image = image.astype(numpy.uint8)
  # make the bounding box square shape by extending the horizontal position by 2 pixels times width/20
  corrected_bounding_box = BoundingBox(topleft = (bounding_box.topleft[0], bounding_box.topleft[1] - bounding_box.size[1] / 10.), size = bounding_box.size)

  for scale in scales:
    bs = corrected_bounding_box.scale(scale, centered=True)
    for y in shifts:
      by = bs.shift((y * bs.size[0], 0))
      for x in shifts:
        bb = by.shift((0, x * bs.size[1]))

        top = max(bb.topleft[0], 0)
        left = int(max(bb.topleft[1], 0))
        bottom = min(bb.bottomright[0], image.shape[0])
        right = int(min(bb.bottomright[1], image.shape[1]))
        landmarks = localizer.locate(uint8_image, top, left, bottom-top, right-left)

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


