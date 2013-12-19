#!ipython


import argparse
import facereclib
import bob
import numpy
import math
import xbob.boosting
import os

from .. import utils, detector, overlapping_detections
from .._features import BoundingBox

def command_line_options(command_line_arguments):

  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument('--test-image', '-i', required=True, help = "Select the image to detect the face in.")
  parser.add_argument('--distance', '-s', type=int, default=2, help = "The distance with which the image should be scanned.")
  parser.add_argument('--scale-base', '-S', type=float, default = math.pow(2.,-1./16.), help = "The logarithmic distance between two scales (should be between 0 and 1).")
  parser.add_argument('--lowest-scale', '-f', type=float, default = 0.125, help = "Faces which will be lower than the given scale times the image resolution will not be found.")
  parser.add_argument('--trained-file', '-r', default = 'detector.hdf5', help = "The file to write the resulting trained detector into.")
  parser.add_argument('--prediction-threshold', '-T', type = float, help = "If given, all detection above this threshold will be displayed.")
  parser.add_argument('--prune-detections', '-p', type=float, help = "If given, detections that overlap with the given threshold are pruned")
  parser.add_argument('--best-detection-overlap', '-b', type=float, help = "If given, the average of the overlapping detections with this minimum overlap will be considered.")
  parser.add_argument('--show-landmarks', '-l', action='store_true', help = "Display the detected landmarks as well?")

  facereclib.utils.add_logger_command_line_option(parser)
  args = parser.parse_args(command_line_arguments)
  facereclib.utils.set_verbosity_level(args.verbose)

  return args


def detect_landmarks(image, bounding_box):
  import xbob.flandmark
  localizer = xbob.flandmark.Localizer()
  scales = [1., 0.9, 0.8, 1.1, 1.2]
  shifts = [0, 0.1, 0.2, -0.1, -0.2]

  uint8_image = image.astype(numpy.uint8)

  for scale in scales:
    bs = bounding_box.scale_centered(scale)
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


def draw_bb(image, bb, color):
  bob.ip.draw_box(image, y=bb.top, x=bb.left, height=bb.bottom - bb.top + 1, width=bb.right - bb.left + 1, color=color)


def main(command_line_arguments = None):
  args = command_line_options(command_line_arguments)

  facereclib.utils.debug("Loading strong classifier from file %s" % args.trained_file)
  # load classifier and feature extractor
  classifier, feature_extractor, is_cpp_extractor, mean, variance = detector.load(args.trained_file)

  sampler = detector.Sampler(distance=args.distance, scale_factor=args.scale_base, lowest_scale=args.lowest_scale, cpp_implementation=is_cpp_extractor)

  # load test file
  test_image = bob.io.load(args.test_image)
  if test_image.ndim == 3:
    test_image = bob.ip.rgb_to_gray(test_image)

  detections = []
  predictions = []
  # get the detection scores for the image
  feature_vector = numpy.zeros(feature_extractor.number_of_features, numpy.uint16)
  for bounding_box in sampler.iterate(test_image, feature_extractor, feature_vector):
    prediction = classifier(feature_vector)
    if args.prediction_threshold is None or prediction > args.prediction_threshold:
      detections.append(bounding_box)
      predictions.append(prediction)
      facereclib.utils.debug("Found bounding box %s with value %f" % (str(bounding_box), prediction))

  # prune detections
  detections, predictions = utils.prune(detections, predictions, args.prune_detections)

  facereclib.utils.info("Number of (pruned) detections: %d" % len(detections))
  highest_detection = predictions[0]
  facereclib.utils.info("Best detection with value %f at %s: " % (highest_detection, str(detections[0])))

  if args.best_detection_overlap is not None:
    # compute average over the best locations
    bb, value = utils.best_detection(detections, predictions, args.best_detection_overlap)
    detections = [bb]
    facereclib.utils.info("Limiting to a single BoundingBox %s with value %f" % (str(detections[0]), value))

  # compute best location
  elif args.prediction_threshold is None:
    # get the detection with the highest value
    detections = detections[:1]
    facereclib.utils.info("Limiting to the best BoundingBox")

  color_image = bob.io.load(args.test_image)
  if color_image.ndim == 2:
    color_image = bob.ip.gray_to_rgb(color_image)
  for detection, prediction in zip(detections, predictions):
    color = (255,0,0) if args.prediction_threshold is None else (int(255. * (prediction - args.prediction_threshold) / (highest_detection-args.prediction_threshold)),0,0)
    draw_bb(color_image, detection, color)

  if len(detections) == 1 and args.show_landmarks:
    landmarks = detect_landmarks(test_image, detections[0])
    facereclib.utils.info("Detected %d landmarks" % (len(landmarks)))
    for i in range(len(landmarks)):
      bob.ip.draw_cross(color_image, y=int(landmarks[i][0]), x=int(landmarks[i][1]), radius=detections[0].height/30, color = (0,255,0) if i else (0,0,255))


  import matplotlib.pyplot as mpl

  rolled = numpy.rollaxis(numpy.rollaxis(color_image, 2),2)

  mpl.imshow(rolled)
  raw_input("Press Enter to continue...")

  bob.io.save(test_image, 'result.png')

