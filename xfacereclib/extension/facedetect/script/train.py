
import argparse
import facereclib
import bob
import numpy
import math
import xbob.boosting
import os

from ..detector import Sampler, LBPFeatures, MBLBPFeatures, save_features
from .. import utils


LBP_CLASSES = {
  'MBLBP' : MBLBPFeatures,
  'LBP'   : LBPFeatures
}

LBP_VARIANTS = {
  'ell'  : {'circular' : True},
  'u2'   : {'uniform' : True},
  'ri'   : {'rotation_invariant' : True},
  'mct'  : {'to_average' : True, 'add_average_bit' : True},
  'tran' : {'elbp_type' : bob.ip.ELBPType.TRANSITIONAL},
  'dir'  : {'elbp_type' : bob.ip.ELBPType.DIRECTION_CODED}
}

def lbp_variant(cls, variants, overlap, scale):
  """Returns the kwargs that are required for the LBP variant."""
  res = {}
  for t in variants:
    res.update(LBP_VARIANTS[t])
  if overlap:
    res['overlap'] = overlap
  if scale:
    if cls == 'MBLBP':
      res['lbp_extractors'] = [bob.ip.LBP(8, block_size=(scale,scale), **res)]
    else:
      res['lbp_extractors'] = [bob.ip.LBP(8, radius=scale, **res)]

  return LBP_CLASSES[cls](**res)


def command_line_options(command_line_arguments):

  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument('--databases', '-d', default=['banca'], nargs='+', help = "Select the databases to get the training images from.")

  parser.add_argument('--lbp-class', '-c', choices=LBP_CLASSES.keys(), default='MBLBP', help = "Specify, which type of LBP features are desired.")
  parser.add_argument('--lbp-variant', '-l', choices=LBP_VARIANTS.keys(), nargs='+', default = [], help = "Specify, which LBP variant(s) are wanted (ell is not available for MBLPB codes).")
  parser.add_argument('--lbp-overlap', '-o', action='store_true', help = "Specify the overlap of the MBLBP.")
  parser.add_argument('--lbp-scale', '-L', type=int, help="If given, only a single LBP extractor with the given LBP scale will be extracted, otherwise all possible scales are generated taken.")

  parser.add_argument('--rounds', '-r', default=10, type=int, help = "The number of training rounds to perform.")
  parser.add_argument('--patch-size', '-p', type=int, nargs=2, default=(24,20), help = "The size of the patch for the image in y and x.")
  parser.add_argument('--distance', '-s', type=int, default=2, help = "The distance with which the image should be scanned.")
  parser.add_argument('--scale-base', '-S', type=float, default = math.sqrt(0.5), help = "The logarithmic distance between two scales (should be between 0 and 1).")
  parser.add_argument('--similarity-thresholds', '-t', type=float, nargs=2, default=(0.3, 0.7), help = "The bounding box overlap thresholds for which negative (< thres[0]) and positive (> thers[1]) examples are accepted.")

  parser.add_argument('--examples-per-image-scale', '-e', type=int, nargs=2, default = [100, 100], help = "The number of positive and negative training examples for each image scale.")
  parser.add_argument('--training-examples', '-E', type=int, nargs=2, default = [10000, 10000], help = "The number of positive and negative training examples to sample.")
  parser.add_argument('--limit-training-files', '-y', type=int, help = "Limit the number of training files (for debug purposes only).")
  parser.add_argument('--write-examples', '-x', help = "Write the positive training examples to the given directory.")

  parser.add_argument('--trained-file', '-w', default = 'detector.hdf5', help = "The file to write the resulting trained detector into.")

  facereclib.utils.add_logger_command_line_option(parser)
  args = parser.parse_args(command_line_arguments)
  facereclib.utils.set_verbosity_level(args.verbose)

  return args



def main(command_line_arguments = None):
  args = command_line_options(command_line_arguments)

  # get training data
  training_files = utils.image_annot(args.databases, args.limit_training_files)

  # create the training set
  sampler = Sampler(patch_size=args.patch_size, scale_factor=args.scale_base, distance=args.distance, similarity_thresholds=args.similarity_thresholds)
  preprocessor = facereclib.preprocessing.NullPreprocessor()

  facereclib.utils.info("Loading %d training images" % len(training_files))
  for file_name, annotations in training_files:
    facereclib.utils.debug("Loading image file '%s' with %d faces" % (file_name, len(annotations)))
    image = preprocessor(preprocessor.read_original_data(file_name))
    try:
      boxes = [utils.BoundingBox(**annotation) for annotation in annotations]
      sampler.add(image, boxes, args.examples_per_image_scale[0], args.examples_per_image_scale[1])
    except KeyError as e:
      facereclib.utils.warn("Ignoring file '%s' since the eye annotations are incomplete" % file_name)

  if args.write_examples:
    facereclib.utils.ensure_dir(args.write_examples)
    sampler._write(os.path.join(args.write_examples, "image_%i.png"))

  # extract all features
  facereclib.utils.info("Extracting features")
  feature_extractor = lbp_variant(args.lbp_class, args.lbp_variant, args.lbp_overlap, args.lbp_scale)
  features, labels = sampler.get(feature_extractor, None, args.training_examples[0], args.training_examples[1])

  # train the boosting algorithm
  facereclib.utils.info("Training machine for %d rounds with %d features" % (args.rounds, labels.shape[0]))
  booster = xbob.boosting.core.boosting.Boost('LutTrainer', args.rounds, feature_extractor.maximum_label())
  boosted_machine = booster.train(features, labels)

  # get the predictions of the training set
  predictions = numpy.ndarray((features.shape[0],), numpy.float64)
  predicted_labels = numpy.ndarray((features.shape[0],), numpy.float64)
  boosted_machine(features, predictions, predicted_labels)
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
