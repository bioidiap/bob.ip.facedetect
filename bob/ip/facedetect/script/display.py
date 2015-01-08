#!ipython


import argparse
import facereclib
import numpy
import math
import os
import pkg_resources

import bob.io.base
import bob.io.image
import bob.ip.color
import bob.ip.draw
#import bob.ip.flandmark

from .. import utils, detector, overlapping_detections
from .._features import BoundingBox, prune_detections

def command_line_options(command_line_arguments):

  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument('test_image', help = "Select the image to detect the face in.")
  parser.add_argument('--distance', '-s', type=int, default=2, help = "The distance with which the image should be scanned.")
  parser.add_argument('--scale-base', '-S', type=float, default = math.pow(2.,-1./16.), help = "The logarithmic distance between two scales (should be between 0 and 1).")
  parser.add_argument('--lowest-scale', '-f', type=float, default = 0.125, help = "Faces which will be lower than the given scale times the image resolution will not be found.")
  parser.add_argument('--cascade-file', '-r', help = "The file to read the resulting cascade from; If left empty, the default cascade will be loaded")
  parser.add_argument('--prediction-threshold', '-t', type = float, help = "If given, all detection above this threshold will be displayed.")
  parser.add_argument('--prune-detections', '-p', type=float, help = "If given, detections that overlap with the given threshold are pruned")
  parser.add_argument('--best-detection-overlap', '-b', type=float, help = "If given, the average of the overlapping detections with this minimum overlap will be considered.")
  parser.add_argument('--write-detection', '-w', help = "If given, the resulting image will be written to the given file.")

  facereclib.utils.add_logger_command_line_option(parser)
  args = parser.parse_args(command_line_arguments)
  facereclib.utils.set_verbosity_level(args.verbose)

  if args.cascade_file is None:
    args.cascade_file = pkg_resources.resource_filename('xfacereclib.extension.facedetect', 'MCT_cascade.hdf5')

  return args


def draw_bb(image, bb, color):
  bob.ip.draw.box(image, bb.topleft, bb.size, color)


def main(command_line_arguments = None):
  args = command_line_options(command_line_arguments)

  facereclib.utils.debug("Loading cascade from file %s" % args.cascade_file)
  # load classifier and feature extractor
  cascade = detector.Cascade(classifier_file=bob.io.base.HDF5File(args.cascade_file))

  sampler = detector.Sampler(distance=args.distance, scale_factor=args.scale_base, lowest_scale=args.lowest_scale)

  # load test file
  test_image = bob.io.base.load(args.test_image)
  if test_image.ndim == 3:
    test_image = bob.ip.color.rgb_to_gray(test_image)

  detections = []
  predictions = []
  # get the detection scores for the image
  for prediction, bounding_box in sampler.iterate_cascade(cascade, test_image):
    if args.prediction_threshold is None or prediction > args.prediction_threshold:
      detections.append(bounding_box)
      predictions.append(prediction)
      facereclib.utils.debug("Found bounding box %s with value %f" % (str(bounding_box), prediction))

  # prune detections
  if args.prune_detections is None:
    detections, predictions = prune_detections(detections, numpy.array(predictions), 1)
  else:
    detections, predictions = prune_detections(detections, numpy.array(predictions), args.prune_detections)


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

  color_image = bob.io.base.load(args.test_image)
  if color_image.ndim == 2:
    color_image = bob.ip.color.gray_to_rgb(color_image)
  for detection, prediction in zip(detections, predictions):
    color = (255,0,0) if args.prediction_threshold is None else (int(255. * (prediction - args.prediction_threshold) / (highest_detection-args.prediction_threshold)),0,0)
    draw_bb(color_image, detection, color)

  import matplotlib.pyplot as mpl

  rolled = numpy.rollaxis(numpy.rollaxis(color_image, 2),2)

  mpl.imshow(rolled)
  raw_input("Press Enter to continue...")

  if args.write_detection:
    bob.io.base.save(color_image, args.write_detection)

