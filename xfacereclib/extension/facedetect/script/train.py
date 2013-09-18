
import argparse
import facereclib
import bob
import numpy
import math
import xbob.boosting

from ..utils import BoundingBox
from ..detector import Sampler, LBPFeatures, MBLBPFeatures, save_features


LBP_CLASSES = {
  'MBLBP' : MBLBPFeatures,
  'LBP'   : LBPFeatures
}

LBP_VARIANTS = {
  'ell'  : {'circular' : True},
  'u2'   : {'uniform' : True},
  'ri'   : {'rotation_invariant' : True},
  'avg'  : {'to_average' : True, 'add_average_bit' : True},
  'tran' : {'elbp_type' : bob.ip.ELBPType.TRANSITIONAL},
  'dir'  : {'elbp_type' : bob.ip.ELBPType.DIRECTION_CODED}
}

def lbp_variant(cls, variants):
  """Returns the kwargs that are required for the LBP variant."""
  res = {}
  for t in variants:
    res.update(LBP_VARIANTS[t])
  return LBP_CLASSES[cls](**res)


def command_line_options(command_line_arguments):

  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument('--database', '-d', default='banca', help = "Select the database to get the training images from.")

  parser.add_argument('--lbp-class', '-c', choices=LBP_CLASSES.keys(), default='MBLBP', help = "Specify, which type of LBP features are desired.")
  parser.add_argument('--lbp-variant', '-l', choices=LBP_VARIANTS.keys(), nargs='+', default = [], help = "Specify, which LBP variant(s) are wanted (ell is not available for MBLPB codes).")

  parser.add_argument('--rounds', '-r', default=10, type=int, help = "The number of training rounds to perform.")
  parser.add_argument('--patch-size', '-p', type=int, nargs=2, default=(24,20), help = "The size of the patch for the image in y and x.")
  parser.add_argument('--distance', '-s', type=int, default=2, help = "The distance with which the image should be scanned.")
  parser.add_argument('--scale-base', '-S', type=float, default = math.sqrt(0.5), help = "The logarithmic distance between two scales (should be between 0 and 1).")
  parser.add_argument('--similarity-thresholds', '-t', type=float, nargs=2, default=(0.3, 0.7), help = "The bounding box overlap thresholds for which negative (< thres[0]) and positive (> thers[1]) examples are accepted.")

  parser.add_argument('--examples-per-image-scale', '-e', type=int, nargs=2, default = [100, 100], help = "The number of positive and negative training examples for each image scale.")
  parser.add_argument('--training-examples', '-E', type=int, nargs=2, default = [10000, 10000], help = "The number of positive and negative training examples to sample.")
  parser.add_argument('--limit-training-files', '-y', type=int, help = "Limit the number of training files (for debug purposes only).")

  parser.add_argument('--trained-file', '-w', default = 'detector.hdf5', help = "The file to write the resulting trained detector into.")

  facereclib.utils.add_logger_command_line_option(parser)
  args = parser.parse_args(command_line_arguments)
  facereclib.utils.set_verbosity_level(args.verbose)

  return args



def main(command_line_arguments = None):
  args = command_line_options(command_line_arguments)

  # open database to collect training images
  database = facereclib.utils.resources.load_resource(args.database, 'database')
  training_files = database.training_files()
  training_files = [training_files[t] for t in facereclib.utils.quasi_random_indices(len(training_files), args.limit_training_files)]

  # create the training set
  sampler = Sampler(patch_size=args.patch_size, scale_factor=args.scale_factor, distance=args.distance, similarity_thresholds=args.similarity_thresholds)
  preprocessor = facereclib.preprocessing.NullPreprocessor()

  facereclib.utils.info("Loading %d training images" % len(training_files))
  for file in training_files:
    facereclib.utils.debug("Loading image file '%s'" % file.path)
    image = preprocessor(preprocessor.read_original_data(str(file.make_path(database.original_directory, database.original_extension))))

    annotations = database.annotations(file)

    sampler.add(image, (BoundingBox(source='eyes', **annotations),), args.examples_per_image_scale[0], args.examples_per_image_scale[1])


  # extract all features
  facereclib.utils.info("Extracting features")
  feature_extractor = lbp_variant(args.lbp_class, args.lbp_variant)
  features, labels = sampler.get(feature_extractor, args.training_examples[0], args.training_examples[1])

  # train the boosting algorithm
  facereclib.utils.info("Training machine for %d rounds with %d features" % (args.rounds, labels.shape[0]))
  booster = xbob.boosting.core.boosting.Boost('LutTrainer', args.rounds, feature_extractor.maximum_label())
  boosted_machine = booster.train(features, labels)

  # get the predictions of the training set
  predictions, predicted_labels = boosted_machine.classify(features)
  decisions = numpy.ones(predictions.shape, numpy.float64)
  decisions[predictions < 0] = -1

  print labels
  print predicted_labels

  # write the machine and the feature extractor into the same HDF5 file
  f = bob.io.HDF5File(args.trained_file, 'w')
  f.create_group("Machine")
  f.create_group("Features")
  f.cd("/Machine")
  boosted_machine.save(f)
  f.cd("/Features")
  save_features(feature_extractor, f)
  del f

  correct = numpy.count_nonzero(labels == predicted_labels)

  print "The number of correctly classified training examples is", correct, "of", predictions.shape[0], "examples"


  # try to classify with the given machine
#  for i in range(len(features)):
 #   print("Predicted label = %d, while real label = %d" % (predictions[i], labels[i]))

#  training_examples._save(positive_only=False)
