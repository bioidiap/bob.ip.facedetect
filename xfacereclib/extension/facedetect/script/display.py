#!ipython


import argparse
import facereclib
import bob
import numpy
import math
import xbob.boosting
import xbob.flandmark
import os
import pkg_resources

from .. import utils, detector, overlapping_detections
from ..graph import FaceGraph, ActiveShapeModel, JetStatistics
from .._features import BoundingBox, prune_detections

def command_line_options(command_line_arguments):

  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument('--test-image', '-i', required=True, help = "Select the image to detect the face in.")
  parser.add_argument('--distance', '-s', type=int, default=2, help = "The distance with which the image should be scanned.")
  parser.add_argument('--scale-base', '-S', type=float, default = math.pow(2.,-1./16.), help = "The logarithmic distance between two scales (should be between 0 and 1).")
  parser.add_argument('--lowest-scale', '-f', type=float, default = 0.125, help = "Faces which will be lower than the given scale times the image resolution will not be found.")
  parser.add_argument('--cascade-file', '-r', help = "The file to write the resulting trained detector into.")
  parser.add_argument('--localizer-file', '-l', help = "The file to get the localizer from")
  parser.add_argument('--local-model-type', '-T', choices = ("graph", "asm", "stat"), help = "Select the type of local model that you want to use instead of flandmark")
  parser.add_argument('--local-model-file', '-R', help = "Use the given local model file instead of flandmark.")
  parser.add_argument('--asm-local-search-region', '-A', type=int, default=15, help = "The region, where local patches are searched in for the ASM algorithm")
  parser.add_argument('--prediction-threshold', '-t', type = float, help = "If given, all detection above this threshold will be displayed.")
  parser.add_argument('--prune-detections', '-p', type=float, help = "If given, detections that overlap with the given threshold are pruned")
  parser.add_argument('--best-detection-overlap', '-b', type=float, help = "If given, the average of the overlapping detections with this minimum overlap will be considered.")

  facereclib.utils.add_logger_command_line_option(parser)
  args = parser.parse_args(command_line_arguments)
  facereclib.utils.set_verbosity_level(args.verbose)

  if args.cascade_file is None:
    args.cascade_file = pkg_resources.resource_filename('xfacereclib.extension.facedetect', 'MCT_cascade.hdf5')

  return args


def draw_bb(image, bb, color):
  bob.ip.draw_box(image, y=bb.top, x=bb.left, height=bb.bottom - bb.top + 1, width=bb.right - bb.left + 1, color=color)


def main(command_line_arguments = None):
  args = command_line_options(command_line_arguments)

  facereclib.utils.debug("Loading cascade from file %s" % args.cascade_file)
  # load classifier and feature extractor
  cascade = detector.Cascade(classifier_file=bob.io.HDF5File(args.cascade_file))

  if args.localizer_file is not None:
    localizer, feature_extractor, _, _ = detector.load(args.localizer_file)

  elif args.local_model_type is not None:
    assert args.local_model_file is not None
    local_model = {
      'graph' : FaceGraph(),
      'stat'  : JetStatistics(),
      'asm'   : ActiveShapeModel(local_search_distance=args.asm_local_search_region)
    }[args.local_model_type]
    local_model.load(bob.io.HDF5File(args.local_model_file))
    facereclib.utils.info("Loading local model of type %s from %s" % (args.local_model_type, args.local_model_file))

  sampler = detector.Sampler(distance=args.distance, scale_factor=args.scale_base, lowest_scale=args.lowest_scale)

  # load test file
  test_image = bob.io.load(args.test_image)
  if test_image.ndim == 3:
    test_image = bob.ip.rgb_to_gray(test_image)

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

  color_image = bob.io.load(args.test_image)
  if color_image.ndim == 2:
    color_image = bob.ip.gray_to_rgb(color_image)
  for detection, prediction in zip(detections, predictions):
    color = (255,0,0) if args.prediction_threshold is None else (int(255. * (prediction - args.prediction_threshold) / (highest_detection-args.prediction_threshold)),0,0)
    draw_bb(color_image, detection, color)

  if len(detections) == 1:
    if args.localizer_file is not None:
#    landmarks = utils.detect_landmarks(xbob.flandmark.Localizer(), test_image, detections[0])
      landmarks = utils.localize(localizer, feature_extractor, test_image, detections[0])
    elif args.local_model_file is not None:
      landmarks = utils.predict(local_model, test_image, detections[0])

    facereclib.utils.info("Detected %d landmarks" % (len(landmarks)))
    for i in range(len(landmarks)):
      bob.ip.draw_cross(color_image, y=int(landmarks[i][0]), x=int(landmarks[i][1]), radius=detections[0].height/30, color = (0,255,0) if i else (0,0,255))


  import matplotlib.pyplot as mpl

  rolled = numpy.rollaxis(numpy.rollaxis(color_image, 2),2)

  mpl.imshow(rolled)
  raw_input("Press Enter to continue...")

  bob.io.save(test_image, 'result.png')

