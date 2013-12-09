
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

  parser.add_argument('--databases', '-d', default=('banca',), nargs="+", help = "Select the database to get the training images from.")
  parser.add_argument('--limit-validation-files', '-y', type=int, help = "Limit the validation files to the given number (for debug purposes mainly)")
  parser.add_argument('--limit-classifiers', '-Y', type=int, help = "Limit the number of classifiers that are evaluated (for debug purposes mainly)")
  parser.add_argument('--kept-negatives', '-l', type=float, nargs='+', default=[0.2, 0.1, 0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001], help = "The relative number of negative values not rejected in the step of the cascade")
  parser.add_argument('--rejection-rates', '-L', type=float, nargs='+', default=[0.05, 0.07, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15], help = "The relative number of **positive** values rejected in the step of the cascade")
  parser.add_argument('--distance', '-s', type=int, default=4, help = "The distance with which the image should be scanned.")
  parser.add_argument('--scale-base', '-S', type=float, default = math.pow(2.,-1./4.), help = "The logarithmic distance between two scales (should be between 0 and 1).")
  parser.add_argument('--first-scale', '-f', type=float, default = 0.5, help = "The first scale of the image to consider (should be between 0 and 1, higher values will slow down the detection process).")
  parser.add_argument('--trained-file', '-r', default = 'detector.hdf5', help = "The file to compute the cascade for.")
  parser.add_argument('--cascade-file', '-w', default = 'cascade.hdf5', help = "The file to write the resulting cascade into.")
#  parser.add_argument('--prune-detections', '-p', type=float, help = "If given, detections that overlap with the given threshold are pruned")
  parser.add_argument('--detection-threshold', '-j', type=float, default=0.7, help = "The overlap from Ground Truth for which a detection should be considered as successful")


  facereclib.utils.add_logger_command_line_option(parser)
  args = parser.parse_args(command_line_arguments)
  facereclib.utils.set_verbosity_level(args.verbose)

  return args

def main(command_line_arguments = None):
  args = command_line_options(command_line_arguments)

  # open database to collect test images
  validation_files = utils.test_image_annot(args.databases, [None for i in range(len(args.databases))], args.limit_validation_files)
  preprocessor = facereclib.preprocessing.NullPreprocessor()

  facereclib.utils.debug("Loading strong classifier from file %s" % args.trained_file)
  # load classifier and feature extractor
  classifier, feature_extractor, is_cpp_extractor, mean, variance = detector.load(args.trained_file)
  feature_vector = numpy.zeros(feature_extractor.number_of_features, numpy.uint16)

  # create the test examples
  sampler = detector.Sampler(distance=args.distance, scale_factor=args.scale_base, first_scale=args.first_scale, cpp_implementation=is_cpp_extractor)

  # generate a DENSE cascade of classifiers
  classifiers = [classifier.__class__(classifier.weak_machines[i:i+1], classifier.weights[i:i+1, 0]) for i in range(args.limit_classifiers if args.limit_classifiers is not None else len(classifier.weak_machines))]

  positives = []
  negatives = []

  # compute cascade values (thresholds, steps)
  last_cascade_index = 0
  cascade_step = 0
  cascade = detector.Cascade(feature_extractor=feature_extractor)

  facereclib.utils.info("Starting validation with %d images" % len(validation_files))
  # iterate over all weak classifiers
  for index, predictor in enumerate(classifiers):
    facereclib.utils.info("Starting evaluation of round %d of %d" % (index+1, len(classifiers)))
    feature_extractor.model_indices = predictor.indices
    # iterate over the validation files and generate values for ALL classifiers in ALL subwindows
    pos_index = neg_index = 0
    for filename, annotations, _ in validation_files:
      # load image
      image = preprocessor(preprocessor.read_original_data(filename))
      # get ground truth bounding boxes from annotations
      if is_cpp_extractor:
        ground_truth = [utils.bounding_box_from_annotation(**annotation) for annotation in annotations]
      else:
        ground_truth = [utils.BoundingBox(**annotation) for annotation in annotations]

      # get the detection scores for the image
      for bounding_box in sampler.iterate(image, feature_extractor, feature_vector):
        # check if the bounding box is a positive
        positive = False
        for gt in ground_truth:
          if gt.similarity(bounding_box) > args.detection_threshold:
            positive = True
            break

        # compute the prediction with the current cascade entry
        prediction = predictor(feature_vector)
        if positive:
          if index: positives[pos_index] += prediction
          else: positives.append(prediction)
          pos_index += 1
        else:
          if index: negatives[neg_index] += prediction
          else: negatives.append(prediction)
          neg_index += 1

    # compute the threshold for the number of accepted false positives
    s = sorted(negatives, reverse=True)
    threshold = s[int(args.kept_negatives[cascade_step] * len(s))]
    del s

    # compute acceptance rate for this threshold
    rejection_rate = float(numpy.count_nonzero(numpy.array(positives) <= threshold)) / float(len(positives))
    if rejection_rate < args.rejection_rates[cascade_step]:
      facereclib.utils.info("Found cascade step after %d extractors with threshold %f and rejection rate %3.2f%%" % (index+1, threshold, rejection_rate * 100.))
      cascade.add(classifier, last_cascade_index, index+1, threshold)
      last_cascade_index = index + 1
      cascade_step += 1
      cascade.save(bob.io.HDF5File("cascade_index_%d.hdf5" % cascade_step, 'w'))

      if cascade_step >= len(args.kept_negatives):
        # we have used all our acceptance rates, so we can stop here
        break
    else:
      facereclib.utils.info("Rejection rate for cascade step %d with threshold %f is too high: %3.2f%%" % (index+1, threshold, rejection_rate * 100.))

  # add the remaining classifiers, just in case
  cascade.add(classifier, last_cascade_index, len(classifiers), threshold)


  # write the cascade into the cascade file
  hdf5 = bob.io.HDF5File(args.cascade_file, 'w')
  cascade.save(hdf5)
  feature_extractor.save(hdf5)

