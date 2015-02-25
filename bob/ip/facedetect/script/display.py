#!ipython

"""Displays the detected face(s) in a given image.

By default, only a single face (the one with the highest detection score) is displayed.
However, when the --prediction-threshold is specified, all detections with scores > --prediction-threshold are shown (with color intensity relative to the score).
"""

import argparse
import numpy
import math
import os

import bob.io.base
import bob.ip.draw
import pkg_resources

import bob.ip.facedetect
import bob.core
logger = bob.core.log.setup("bob.ip.facedetect")

def command_line_options(command_line_arguments):

  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument('test_image', help = "Select the image to detect the face in.")
  parser.add_argument('--distance', '-s', type=int, default=2, help = "The distance with which the image should be scanned.")
  parser.add_argument('--scale-base', '-S', type=float, default = math.pow(2.,-1./16.), help = "The logarithmic distance between two scales (should be between 0 and 1).")
  parser.add_argument('--lowest-scale', '-f', type=float, default = 0.125, help = "Faces which will be lower than the given scale times the image resolution will not be found.")
  parser.add_argument('--cascade-file', '-r', default = pkg_resources.resource_filename('bob.ip.facedetect', 'MCT_cascade.hdf5'), help = "The file to read the resulting cascade from; If left empty, the default cascade will be loaded")
  parser.add_argument('--prediction-threshold', '-t', type = float, help = "If given, all detection above this threshold will be displayed.")
  parser.add_argument('--prune-detections', '-p', type=float, default=1., help = "If given, detections that overlap with the given threshold are pruned")
  parser.add_argument('--best-detection-overlap', '-b', type=float, help = "If given, the average of the overlapping detections with this minimum overlap will be considered.")
  parser.add_argument('--write-detection', '-w', help = "If given, the resulting image will be written to the given file.")

  bob.core.log.add_command_line_option(parser)
  args = parser.parse_args(command_line_arguments)
  bob.core.log.set_verbosity_level(logger, args.verbose)

  return args


def main(command_line_arguments = None):
  args = command_line_options(command_line_arguments)

  logger.debug("Loading cascade from file %s", args.cascade_file)
  # load classifier and feature extractor
  cascade = bob.ip.facedetect.detector.Cascade(bob.io.base.HDF5File(args.cascade_file))

  sampler = bob.ip.facedetect.detector.Sampler(distance=args.distance, scale_factor=args.scale_base, lowest_scale=args.lowest_scale)

  # load test file
  test_image = bob.io.base.load(args.test_image)
  if test_image.ndim == 3:
    test_image = bob.ip.color.rgb_to_gray(test_image)

  detections = []
  predictions = []
  # get the detection scores for the image
  for prediction, bounding_box in sampler.iterate_cascade(cascade, test_image, args.prediction_threshold):
    detections.append(bounding_box)
    predictions.append(prediction)
    logger.debug("Found bounding box %s with value %f", str(bounding_box), prediction)

  # prune detections
  detections, predictions = bob.ip.facedetect.prune_detections(detections, numpy.array(predictions), args.prune_detections)


  logger.info("Number of (pruned) detections: %d", len(detections))
  highest_detection = predictions[0]
  logger.info("Best detection with value %f at %s: ", highest_detection, str(detections[0]))

  if args.best_detection_overlap is not None:
    # compute average over the best locations
    bb, value = bob.ip.facedetect.best_detection(detections, predictions, args.best_detection_overlap)
    detections = [bb]
    logger.info("Limiting to a single BoundingBox %s with value %f", str(detections[0]), value)

  # compute best location
  elif args.prediction_threshold is None:
    # get the detection with the highest value
    detections = detections[:1]
    logger.info("Limiting to the best BoundingBox")

  color_image = bob.io.base.load(args.test_image)
  if color_image.ndim == 2:
    color_image = bob.ip.color.gray_to_rgb(color_image)
  for detection, prediction in zip(detections, predictions):
    color = (255,0,0) if args.prediction_threshold is None else (int(255. * (prediction - args.prediction_threshold) / (highest_detection-args.prediction_threshold)),0,0)
    bob.ip.draw.box(color_image, detection.topleft, detection.size, color)

  import matplotlib.pyplot as mpl

  rolled = numpy.rollaxis(numpy.rollaxis(color_image, 2),2)

  mpl.imshow(rolled)
  raw_input("Press Enter to continue...")

  if args.write_detection:
    bob.io.base.save(color_image, args.write_detection)
