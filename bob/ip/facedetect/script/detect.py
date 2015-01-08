
import argparse
import facereclib
import numpy
import math
import os

from .. import utils, detector
from .._features import prune_detections, FeatureExtractor

import bob.io.base

def command_line_options(command_line_arguments):

  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument('--database', '-d', default = 'banca', help = "Select the database to get the training images from.")
  parser.add_argument('--protocol', '-P', help = "If given, the test files from the given protocol are detected.")
  parser.add_argument('--limit-test-files', '-y', type=int, help = "Limit the test files to the given number (for debug purposes mainly)")
  parser.add_argument('--distance', '-s', type=int, default=2, help = "The distance with which the image should be scanned.")
  parser.add_argument('--scale-base', '-S', type=float, default = math.pow(2.,-1./16.), help = "The logarithmic distance between two scales (should be between 0 and 1).")
  parser.add_argument('--lowest-scale', '-f', type=float, default = 0.125, help = "Faces which will be lower than the given scale times the image resolution will not be found.")
  parser.add_argument('--cascade-file', '-r', help = "The file to read the cascade from (has a proper default).")
  parser.add_argument('--prediction-threshold', '-T', type=float, help = "Detections with values below this threshold will be rejected by the detector.")
  parser.add_argument('--score-file', '-w', default='cascaded_scores.txt', help = "The score file to be written.")
  parser.add_argument('--prune-detections', '-p', type=float, default = 1., help = "If given, detections that overlap with the given threshold are pruned")
  parser.add_argument('--detection-threshold', '-j', type=float, default=0.5, help = "The overlap from Ground Truth for which a detection should be considered as successful")

  facereclib.utils.add_logger_command_line_option(parser)
  args = parser.parse_args(command_line_arguments)
  facereclib.utils.set_verbosity_level(args.verbose)

  if args.cascade_file is None:
    import pkg_resources
    args.cascade_file = pkg_resources.resource_filename('xfacereclib.extension.facedetect', 'MCT_cascade.hdf5')

  return args


def main(command_line_arguments = None):
  args = command_line_options(command_line_arguments)

  # open database to collect test images
  facereclib.utils.info("Loading test files...")
  test_files = utils.test_image_annot([args.database], [args.protocol], args.limit_test_files)

  # load cascade
  facereclib.utils.info("Loading cascade from file %s" % args.cascade_file)
  cascade = detector.Cascade(classifier_file=bob.io.base.HDF5File(args.cascade_file))

  # create the test examples
  preprocessor = facereclib.preprocessing.NullPreprocessor()
  sampler = detector.Sampler(distance=args.distance, scale_factor=args.scale_base, lowest_scale=args.lowest_scale)

  # iterate over the test files and detect the faces
  i = 1
  with open(args.score_file, 'w') as f:
    # write configuration
    f.write("# --cascade-file %s --distance %d --scale-base %f --lowest-scale %s --prediction-threshold %s --detection-threshold %f\n" % (args.cascade_file, args.distance, args.scale_base, args.lowest_scale, "None" if args.prediction_threshold is None else "%f" % args.prediction_threshold, args.detection_threshold))
    for filename, annotations, file in test_files:
      facereclib.utils.info("Loading image %d of %d from file '%s'" % (i, len(test_files), filename))
      i += 1
      image = preprocessor(preprocessor.read_original_data(filename))

      # get the detection scores for the image
      predictions = []
      detections = []
      for prediction, bounding_box in sampler.iterate_cascade(cascade, image):
        if args.prediction_threshold is None or prediction > args.prediction_threshold:
          predictions.append(prediction)
          detections.append(bounding_box)

      facereclib.utils.info("Number of detections: %d" % len(detections))

      # prune detections
      detections, predictions = prune_detections(detections, numpy.array(predictions), args.prune_detections)
      facereclib.utils.info("Number of pruned detections: %d" % len(detections))

      # get ground truth bounding boxes from annotations
      ground_truth = [utils.bounding_box_from_annotation(**annotation) for annotation in annotations]
      f.write("%s %d\n" % (file.path, len(ground_truth)))

      # check if we have found a bounding box
      all_positives = []
      for bounding_box in ground_truth:
        for value, detection in zip(predictions, detections):
          if detection.similarity(bounding_box) > args.detection_threshold:
            f.write("%f " % value)
            all_positives.append(detection)
        f.write("\n")
      # write all others as negatives
      for value, detection in zip(predictions, detections):
        if detection not in all_positives:
          f.write("%f " % value)
      f.write("\n")

