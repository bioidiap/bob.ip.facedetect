#!ipython


import argparse
import facereclib
import bob
import numpy
import math
import xbob.boosting
import os

from .. import utils, detector

def command_line_options(command_line_arguments):

  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument('--test-image', '-i', required=True, help = "Select the image to detect the face in.")
  parser.add_argument('--distance', '-s', type=int, default=2, help = "The distance with which the image should be scanned.")
  parser.add_argument('--scale-base', '-S', type=float, default = math.pow(2.,-1./16.), help = "The logarithmic distance between two scales (should be between 0 and 1).")
  parser.add_argument('--first-scale', '-f', type=float, default = 0.5, help = "The first scale of the image to consider (should be between 0 and 1, higher values will slow down the detection process).")
  parser.add_argument('--trained-file', '-r', default = 'detector.hdf5', help = "The file to write the resulting trained detector into.")
  parser.add_argument('--prediction-threshold', '-t', type = float, help = "If given, all detection above this threshold will be displayed.")
  parser.add_argument('--prune-detections', '-p', type=float, help = "If given, detections that overlap with the given threshold are pruned")

  facereclib.utils.add_logger_command_line_option(parser)
  args = parser.parse_args(command_line_arguments)
  facereclib.utils.set_verbosity_level(args.verbose)

  return args

#def classify(classifier, features):
#  return classifier(features)

def draw_bb(image, bb, color):
  bob.ip.draw_box(image, y=bb.top, x=bb.left, height=bb.bottom - bb.top + 1, width=bb.right - bb.left + 1, color=color)

def main(command_line_arguments = None):
  args = command_line_options(command_line_arguments)

  facereclib.utils.debug("Loading strong classifier from file %s" % args.trained_file)
  # load classifier and feature extractor
  classifier, feature_extractor, is_cpp_extractor, mean, variance = detector.load(args.trained_file)

  sampler = detector.Sampler(distance=args.distance, scale_factor=args.scale_base, first_scale=args.first_scale, cpp_implementation=is_cpp_extractor)

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

  if args.prediction_threshold is None:
    detections = detections[:1]

  test_image = bob.ip.gray_to_rgb(test_image)
  for detection, prediction in zip(detections, predictions):
    color = (255,0,0) if args.prediction_threshold is None else (int(255. * (prediction - args.prediction_threshold) / (highest_detection-args.prediction_threshold)),0,0)
    draw_bb(test_image, detection, color)

  import matplotlib.pyplot as mpl

  rolled = numpy.rollaxis(numpy.rollaxis(test_image, 2),2)

  mpl.imshow(rolled)
  raw_input("Press Enter to continue...")

  bob.io.save(test_image, 'result.png')

