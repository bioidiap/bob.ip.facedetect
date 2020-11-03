#!/usr/bin/env python

"""The test script to perform the binary classification on the digits from the MNIST dataset.
The MNIST data is exported using a module similar to the xbob.db.mnist module which provide the train and test partitions for the digits.
Pixel values of grey scale images are used as features and the available algorithms for classification are Lut based Boosting and Stump based Boosting.
Thus it conducts only one binary classifcation test.


"""
from __future__ import print_function

import numpy
import argparse
import os

import bob.io.base
import bob.learn.boosting
import bob.learn.boosting.utils

import bob.core
logger = bob.core.log.setup('bob.learn.boosting')

TRAINER = {
  'stump' : bob.learn.boosting.StumpTrainer,
  'lut'   : bob.learn.boosting.LUTTrainer,
}

LOSS = {
  'exp'   : bob.learn.boosting.ExponentialLoss,
  'log'   : bob.learn.boosting.LogitLoss,
  'tan'   : bob.learn.boosting.TangentialLoss,
}

def command_line_arguments(command_line_options):
  """Defines the command line options."""
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('-t', '--trainer-type', default = 'stump', choices = TRAINER.keys(), help = "The type of weak trainer used for boosting." )
  parser.add_argument('-l', '--loss-type', choices = LOSS.keys(), help = "The type of loss function used in boosting to compute the weights for the weak classifiers.")
  parser.add_argument('-r', '--number-of-boosting-rounds', type = int, default = 100, help = "The number of boosting rounds, i.e., the number of weak classifiers.")

  parser.add_argument('-m', '--multi-variate', action = 'store_true', help = "Perform multi-variate training?")
  parser.add_argument('-s', '--feature-selection-style', default = 'independent', choices = ('independent', 'shared'), help = "The feature selection style (only for multivariate classification with the LUT trainer).")

  parser.add_argument('-d', '--digits', type = int, nargs="+", choices=range(10), default=[5,6], help = "Select the digits you want to compare.")
  parser.add_argument('-a', '--all-digits', action='store_true', help = "Use all digits")
  parser.add_argument('-n', '--number-of-elements', type = int, help = "For testing purposes: limit the number of training and test examples for each class.")
  parser.add_argument('-c', '--classifier-file', help = "If selected, the strong classifier will be stored in this file (or loaded from it if it already exists).")
  parser.add_argument('-F', '--force', action='store_true', help = "Re-train the strong classifier, even if the --classifier-file already exists.")

  bob.core.log.add_command_line_option(parser)
  args = parser.parse_args(command_line_options)
  bob.core.log.set_verbosity_level(logger, args.verbose)

  if args.trainer_type == 'stump' and args.multi_variate:
    raise ValueError("The stump trainer cannot handle multi-variate training.")

  if args.all_digits:
    args.digits = range(10)
  if len(args.digits) < 2:
    raise ValueError("Please select at least two digits to classify, or --all to classify all digits")
  if args.loss_type is None:
    args.loss_type = 'exp' if args.trainer_type == 'stump' else 'log'

  return args


def align(input, output, digits, multi_variate = False):
  if multi_variate:
    # just one classifier, with multi-variate output
    input = numpy.vstack(input).astype(numpy.uint16)
    # create output data
    target = - numpy.ones((input.shape[0], len(output)))
    output = numpy.hstack(output)
    for i,d in enumerate(digits):
      target[output == d, i] = 1
    return {'multi' : (input, target)}

  else:
    # create pairs of one-to-one classifiers
    problems = {}
    for i, d1 in enumerate(digits):
      for j, d2 in enumerate(digits[i+1:]):
        key = "%d-vs-%d" % (d1, d2)
        cur_input = numpy.vstack([input[i], input[j+1]]).astype(numpy.uint16)
        target = numpy.ones((cur_input.shape[0]))
        target[output[i].shape[0]:target.shape[0]] = -1
        problems[key] = (cur_input, target)
    return problems


def read_data(db, which, digits, count, multi_variate):
  input = []
  output = []
  for d in digits:
    digit_data = db.data(which, labels = d)
    if count is not None:
      digit_data = (digit_data[0][:count], digit_data[1][:count])
    input.append(digit_data[0])
    output.append(digit_data[1])

  return align(input, output, digits, multi_variate)


def performance(targets, labels, key, multi_variate):
    difference = targets == labels

    if multi_variate:
      sum = numpy.sum(difference, 1)
      print ("Classified", numpy.sum(sum == difference.shape[1]), "of", difference.shape[0], "elements correctly")
      accuracy = float(numpy.sum(sum == difference.shape[1])) / difference.shape[0]
    else:
      print ("Classified", numpy.sum(difference), "of", difference.shape[0], "elements correctly")
      accuracy = float(numpy.sum(difference)) / difference.shape[0]

    print ("The classification accuracy for", key, "is", accuracy * 100, "%")


def main(command_line_options = None):

  args = command_line_arguments(command_line_options)

  # open (fake) connection to the MNIST database
  db = bob.learn.boosting.utils.MNIST()

  # perform training, if desired
  if args.force and os.path.exists(args.classifier_file):
    os.remove(args.classifier_file)
  if args.classifier_file is None or not os.path.exists(args.classifier_file):
    # get the (aligned) training data
    logger.info("Reading training data")
    training_data = read_data(db, "train", args.digits, args.number_of_elements, args.multi_variate)

    # get weak trainer according to command line options
    if args.trainer_type == 'stump':
      weak_trainer = bob.learn.boosting.StumpTrainer()
    elif args.trainer_type == 'lut':
      weak_trainer = bob.learn.boosting.LUTTrainer(
            256,
            list(training_data.values())[0][1].shape[1] if args.multi_variate else 1,
            args.feature_selection_style
      )
    # get the loss function
    loss_function = LOSS[args.loss_type]()

    # create strong trainer
    trainer = bob.learn.boosting.Boosting(weak_trainer, loss_function)

    strong_classifiers = {}
    for key in sorted(training_data.keys()):
      training_input, training_target = training_data[key]

      if args.multi_variate:
        logger.info("Starting training with %d training samples and %d outputs" % (training_target.shape[0], training_target.shape[1]))
      else:
        logger.info("Starting training with %d training samples for %s" % (training_target.shape[0], key))

      # and train the strong classifier
      strong_classifier = trainer.train(training_input, training_target, args.number_of_boosting_rounds)

      # write strong classifier to file
      if args.classifier_file is not None:
        hdf5 = bob.io.base.HDF5File(args.classifier_file, 'a')
        hdf5.create_group(key)
        hdf5.cd(key)
        strong_classifier.save(hdf5)
        del hdf5

      strong_classifiers[key] = strong_classifier

      # compute training performance
      logger.info("Evaluating training data")
      scores = numpy.zeros(training_target.shape)
      labels = numpy.zeros(training_target.shape)
      strong_classifier(training_input, scores, labels)
      performance(training_target, labels, key, args.multi_variate)

  else:
    # read strong classifier from file
    strong_classifiers = {}
    hdf5 = bob.io.base.HDF5File(args.classifier_file, 'r')
    for key in hdf5.sub_groups(relative=True, recursive=False):
      hdf5.cd(key)
      strong_classifiers[key] = bob.learn.boosting.BoostedMachine(hdf5)
      hdf5.cd("..")

  logger.info("Reading test data")
  test_data = read_data(db, "test", args.digits, args.number_of_elements, args.multi_variate)

  for key in sorted(test_data.keys()):
    test_input, test_target = test_data[key]

    logger.info("Classifying %d test samples for %s" % (test_target.shape[0], key))

    # classify test samples
    scores = numpy.zeros(test_target.shape)
    labels = numpy.zeros(test_target.shape)
    strong_classifiers[key](test_input, scores, labels)

    performance(test_target, labels, key, args.multi_variate)



if __name__ == "__main__":
   main()
