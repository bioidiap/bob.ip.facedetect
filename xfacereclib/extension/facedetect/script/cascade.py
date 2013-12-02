
import argparse
import facereclib
import bob
import numpy
import math
import xbob.boosting
import os

from .. import utils, detector
from .._features import prune_detections

def command_line_options(command_line_arguments):

  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument('--database', '-d', default = 'banca', help = "Select the database to get the training images from.")
  parser.add_argument('--protocol', '-P', help = "If given, the test files from the given protocol are detected.")
  parser.add_argument('--limit-test-files', '-y', type=int, help = "Limit the test files to the given number (for debug purposes mainly)")
  parser.add_argument('--distance', '-s', type=int, default=4, help = "The distance with which the image should be scanned.")
  parser.add_argument('--scale-base', '-S', type=float, default = math.pow(2.,-1./4.), help = "The logarithmic distance between two scales (should be between 0 and 1).")
  parser.add_argument('--first-scale', '-f', type=float, default = 0.5, help = "The first scale of the image to consider (should be between 0 and 1, higher values will slow down the detection process).")
  parser.add_argument('--trained-file', '-r', default = 'detector.hdf5', help = "The file to read the trained detector from.")
  parser.add_argument('--classifiers-per-round', '-n', type=int, default=25, help = "The number of classifiers that should be applied in each cascade.")
  parser.add_argument('--limit-by-variance', '-V', action='store_false', help = "Disable the mean/variance limitation.")
  parser.add_argument('--cascade-threshold', '-t', type=float, default=0., help = "Detections with values below this threshold will be discarded in each round.")
  parser.add_argument('--prediction-threshold', '-T', type=float, help = "Detections with values below this threshold will be rejected by by the complete detector.")
  parser.add_argument('--score-file', '-w', default='cascaded_scores.txt', help = "The score file to be written.")
  parser.add_argument('--prune-detections', '-p', type=float, help = "If given, detections that overlap with the given threshold are pruned")
  parser.add_argument('--detection-threshold', '-j', type=float, default=0.5, help = "The overlap from Ground Truth for which a detection should be considered as successful")


  facereclib.utils.add_logger_command_line_option(parser)
  args = parser.parse_args(command_line_arguments)
  facereclib.utils.set_verbosity_level(args.verbose)

  return args

#def classify(classifier, features):
#  return classifier(features)

def main(command_line_arguments = None):
  args = command_line_options(command_line_arguments)

  # open database to collect test images
  test_files = utils.test_image_annot([args.database], [args.protocol], args.limit_test_files)

  facereclib.utils.debug("Loading strong classifier from file %s" % args.trained_file)
  # load classifier and feature extractor
  classifier, feature_extractor, is_cpp_extractor, mean, variance = detector.load(args.trained_file)
  feature_vector = numpy.zeros(feature_extractor.number_of_features, numpy.uint16)

  if not args.limit_by_variance:
    mean, variance = None, None

  # split the classifier and the feature extractor into cascades
  indices = range(0, len(classifier.weak_machines), args.classifiers_per_round)
  if indices[-1] != len(classifier.weak_machines): indices.append(len(classifier.weak_machines))
  cascade = []
  model_indices = []
  for i in range(len(indices)-1):
    cascade.append(classifier.__class__(classifier.weak_machines[indices[i]:indices[i+1]], classifier.weights[indices[i]:indices[i+1], 0]))
    model_indices.append(classifier.feature_indices(indices[i], indices[i+1]))

  # create the test examples
  preprocessor = facereclib.preprocessing.NullPreprocessor()
  sampler = detector.Sampler(distance=args.distance, scale_factor=args.scale_base, first_scale=args.first_scale, cpp_implementation=is_cpp_extractor)

  # iterate over the test files and detect the faces
  i = 1
  with open(args.score_file, 'w') as f:
    # write configuration
    f.write("# --trained-file %s --distance %d --scale-base %f --first-scale %s --cascade-threshold %f --prediction-threshold %s --detection-threshold %f\n" % (args.trained_file, args.distance, args.scale_base, args.first_scale, args.cascade_threshold, "None" if args.prediction_threshold is None else "%f" % args.prediction_threshold, args.detection_threshold))
    for filename, annotations, file in test_files:
      facereclib.utils.info("Loading image %d of %d from file '%s'" % (i, len(test_files), filename))
      i += 1
      image = preprocessor(preprocessor.read_original_data(filename))

      # get the detection scores for the image
      predictions = []
      detections = []
      for prediction, bounding_box in sampler.cascade(image, feature_extractor, feature_vector, cascade, model_indices, threshold = args.cascade_threshold, mean = mean, variance = variance):
        if args.prediction_threshold is None or prediction > args.prediction_threshold:
          predictions.append(prediction)
          detections.append(bounding_box)

      facereclib.utils.info("Number of detections: %d" % len(detections))

      # prune detections
      detections, predictions = utils.prune(detections, numpy.array(predictions), args.prune_detections)
      facereclib.utils.info("Number of pruned detections: %d" % len(detections))

      # get ground truth bounding boxes from annotations
      if is_cpp_extractor:
        ground_truth = [utils.bounding_box_from_annotation(**annotation) for annotation in annotations]
      else:
        ground_truth = [utils.BoundingBox(**annotation) for annotation in annotations]
      f.write("%s %d\n" % (file.path, len(ground_truth)))

      # check if we have found a bounding box
      all_positives = set()
      for bounding_box in ground_truth:
        for value, detection in zip(predictions, detections):
          if detection.similarity(bounding_box) > args.detection_threshold:
            f.write("%f " % value)
            all_positives.add(detection)
        f.write("\n")
      # write all others as negatives
      for value, detection in zip(predictions, detections):
        if detection not in all_positives:
          f.write("%f " % value)
      f.write("\n")

