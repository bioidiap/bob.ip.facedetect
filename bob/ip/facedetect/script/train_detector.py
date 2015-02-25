"""Trains a cascade of classifiers from the given training features"""

import argparse
import numpy
import math
import os, sys


import bob.ip.base
import bob.learn.boosting
import bob.ip.facedetect

import bob.core
logger = bob.core.log.setup('bob.ip.facedetect')


def command_line_options(command_line_arguments):

  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument('--feature-directory', '-d', default = "features")

  parser.add_argument('--features-in-first-round', '-r', default=8, type=int, help = "The number of features to extract in the first bootstrapping round (will be doubled in each bootstrapping round).")
  parser.add_argument('--bootstrapping-rounds', '-R', default=7, type=int, help = "The number of bootstrapping rounds to perform.")
  parser.add_argument('--training-examples', '-E', type=int, nargs=2, default = [5000, 5000], help = "The number of positive and negative training examples to add per round.")

  parser.add_argument('--classifiers-per-round', '-n', type=int, default=25, help = "The number of classifiers that should be applied the regular cascade.")
  parser.add_argument('--cascade-threshold', '-t', type=float, default=-5, help = "Detections with values below this threshold will be discarded in each round in the regular cascade.")

  parser.add_argument('--force', '-F', action='store_true', help = "Force the re-creation of intermediate files.")

  parser.add_argument('--trained-file', '-w', default = 'cascade.hdf5', help = "The file to write the resulting trained detector into.")

  bob.core.log.add_command_line_option(parser)
  args = parser.parse_args(command_line_arguments)
  bob.core.log.set_verbosity_level(logger, args.verbose)

  return args


def main(command_line_arguments = None):
  args = command_line_options(command_line_arguments)

  train_set = bob.ip.facedetect.train.TrainingSet(feature_directory = args.feature_directory)
  feature_extractor = train_set.feature_extractor()

  # create a boosting trainer
  weak_trainer = bob.learn.boosting.LUTTrainer(feature_extractor.number_of_labels, 1)
  trainer = bob.learn.boosting.Boosting(weak_trainer, bob.learn.boosting.LogitLoss())

  # create a bootstrapper
  bootstrap = bob.ip.facedetect.train.Bootstrap(number_of_rounds=args.bootstrapping_rounds, number_of_weak_learners_in_first_round=args.features_in_first_round, number_of_positive_examples_per_round=args.training_examples[0], number_of_negative_examples_per_round=args.training_examples[1])

  # perform the bootstrapping
  classifier = bootstrap.run(train_set, trainer, filename=args.trained_file, force=args.force)

  logger.info("Creating regular cascade from strong classifier")
  # load classifier and feature extractor
  cascade = bob.ip.facedetect.detector.Cascade(feature_extractor=feature_extractor)
  cascade.create_from_boosted_machine(classifier, classifiers_per_round=args.classifiers_per_round, classification_thresholds=args.cascade_threshold)

  # write the machine and the feature extractor into the same HDF5 file
  cascade.save(bob.io.base.HDF5File(args.trained_file, 'w'))
  logger.info("Saved bootstrapped classifier to file '%s'", args.trained_file)
