
import argparse
import numpy
import math
import os

import bob.io.base
import bob.ip.facedetect
import bob.blitz

import bob.core
logger = bob.core.log.setup("bob.ip.facedetect")

def command_line_options(command_line_arguments):

  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument('--classifiers-per-round', '-n', type=int, default=25, help = "The number of classifiers that should be applied to the regular cascade.")
  parser.add_argument('--cascade-threshold', '-t', type=float, default=-5, help = "Detections with values below this threshold will be discarded in each round in the regular cascade.")

  parser.add_argument('--file-lists', '-i', nargs='+', help = "Select the training lists to extract features for.")
  parser.add_argument('--limit-classifiers', '-Y', type=int, help = "Limit the number of classifiers that are evaluated (for debug purposes mainly)")
  parser.add_argument('--limit-validation-files', '-y', type=int, help = "Limit the validation files to the given number (for debug purposes mainly)")
  parser.add_argument('--kept-negatives', '-l', type=float, nargs='+', default=[0.3, 0.1, 0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001], help = "The relative number of negative values **not** rejected in the step of the cascade")
  parser.add_argument('--rejection-rate', '-L', type=float, default=0.15, help = "The relative number of **positive** values rejected by the cascade")
  parser.add_argument('--distance', '-s', type=int, default=4, help = "The distance with which the image should be scanned.")
  parser.add_argument('--scale-base', '-S', type=float, default = math.pow(2.,-1./8.), help = "The logarithmic distance between two scales (should be between 0 and 1).")
  parser.add_argument('--lowest-scale', '-f', type=float, default=0.125, help = "Patches which will be lower than the given scale times the image resolution will not be taken into account; if 0. (the default) all patches will be considered.")
  parser.add_argument('--input-cascade', '-r', default = 'cascade.hdf5', help = "The file to compute the cascade for.")
  parser.add_argument('--output-cascade', '-w', default = 'cascade.hdf5', help = "The file to write the resulting cascade into.")
#  parser.add_argument('--prune-detections', '-p', type=float, help = "If given, detections that overlap with the given threshold are pruned")
  parser.add_argument('--detection-threshold', '-j', type=float, default=0.7, help = "The overlap from Ground Truth for which a detection should be considered as successful")


  bob.core.log.add_command_line_option(parser)
  args = parser.parse_args(command_line_arguments)
  bob.core.log.set_verbosity_level(logger, args.verbose)

  return args

def main(command_line_arguments = None):
  args = command_line_options(command_line_arguments)

  logger.debug("Loading cascade from file %s", args.input_cascade)
  input_cascade = bob.ip.facedetect.detector.Cascade(bob.io.base.HDF5File(args.input_cascade))

  # get a single strong classifier from the cascade
  strong_classifier = input_cascade.generate_boosted_machine()
  feature_extractor = input_cascade.extractor

  if args.file_lists is None:

    logger.info("Creating regular cascade from strong classifier")
    # load classifier and feature extractor
    cascade = bob.ip.facedetect.detector.Cascade()
    cascade.create_from_boosted_machine(strong_classifier, classifiers_per_round=args.classifiers_per_round, classification_thresholds=args.cascade_threshold, feature_extractor=feature_extractor)

  else:

    # generate and load training set
    train_set = bob.ip.facedetect.train.TrainingSet(feature_directory=None)
    for file_list in args.file_lists:
      logger.info("Loading file list %s", file_list)
      train_set.load(file_list)

    # create feature
    feature_vector = bob.blitz.array((feature_extractor.number_of_features,), numpy.uint16)

    # create the test examples
    sampler = bob.ip.facedetect.detector.Sampler(patch_size= feature_extractor.patch_size, distance=args.distance, scale_factor=args.scale_base, lowest_scale=args.lowest_scale)

    # generate a DENSE cascade of classifiers
    number_of_weak_classifiers = args.limit_classifiers if args.limit_classifiers is not None else len(strong_classifier.weak_machines)
    classifiers = []
    for i in range(number_of_weak_classifiers):
      classifier = bob.learn.boosting.BoostedMachine()
      classifier.add_weak_machine(strong_classifier.weak_machines[i], strong_classifier.weights[i, 0])
      classifiers.append(classifier)

    positives = []
    negatives = []

    # compute cascade values (thresholds, steps)
    last_cascade_index = 0
    cascade_step = 0
    cascade = bob.ip.facedetect.detector.Cascade(feature_extractor=feature_extractor)

    logger.info("Starting validation using %d files", args.limit_validation_files if args.limit_validation_files is not None else len(train_set))
    # iterate over all weak classifiers
    for index, weak_classifier in enumerate(classifiers):
      logger.debug("Starting evaluation of round %d of %d", index+1, len(classifiers))
      feature_extractor.model_indices = weak_classifier.indices
      # iterate over the validation files and generate values for ALL classifiers in ALL subwindows
      pos_index = neg_index = 0
      image_index = 1
      for image, ground_truth, file_name in train_set.iterate(args.limit_validation_files):
#        logger.debug("Processing image %d of %d", image_index, args.limit_validation_files if args.limit_validation_files is not None else len(train_set))
        image_index += 1
        # get the detection scores for the image
        for bounding_box in sampler.iterate(image, feature_extractor, feature_vector):
          # check if the bounding box is a positive
          positive = False
          for gt in ground_truth:
            if gt.similarity(bounding_box) > args.detection_threshold:
              positive = True
              break

          # compute the prediction with the current cascade entry
          prediction = weak_classifier(feature_vector)
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
#      import pdb; pdb.set_trace()
      threshold = s[int(args.kept_negatives[cascade_step] * len(s))]
      del s

      # compute rejection rate of false negatives for this threshold
      rejection_rate = float(numpy.count_nonzero(numpy.array(positives) <= threshold)) / float(len(positives))

      if rejection_rate < args.rejection_rate:
        logger.info("Found cascade step %d after %d extractors with threshold %f and rejection rate %3.2f%%", cascade_step+1, index+1, threshold, rejection_rate * 100.)
        cascade.add(strong_classifier, last_cascade_index, index+1, threshold)
        last_cascade_index = index + 1
        cascade_step += 1
        cascade.save(bob.io.base.HDF5File("cascade_index_%d.hdf5" % cascade_step, 'w'))

        if cascade_step >= len(args.kept_negatives):
          # we have used all our acceptance rates, so we can stop here
          break
      else:
        logger.debug("Rejection rate for cascade step %d with threshold %f is too high: %3.2f%% > %3.2f%%", index+1, threshold, rejection_rate * 100., args.rejection_rate * 100)

    # add the remaining classifiers, just in case
    cascade.add(strong_classifier, last_cascade_index, len(classifiers), threshold)


  # write the cascade into the cascade file
  logger.info("Writing cascade file %s", args.output_cascade)
  hdf5 = bob.io.base.HDF5File(args.output_cascade, 'w')
  cascade.save(hdf5)
