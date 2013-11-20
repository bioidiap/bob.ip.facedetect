
import argparse
import facereclib
import bob
import numpy
import math
import xbob.boosting
import os, sys

from ..detector import Sampler, LBPFeatures, MBLBPFeatures, Bootstrap, save
from .. import BoundingBox, FeatureExtractor, utils


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

def lbp_variant(cls, variants, overlap, scale, square, cpp):
  """Returns the kwargs that are required for the LBP variant."""
  res = {}
  for t in variants:
    res.update(LBP_VARIANTS[t])
  if cpp:
    if scale:
      if cls == 'MBLBP':
        return FeatureExtractor(patch_size = (24,20), extractors = [bob.ip.LBP(8, block_size=(scale,scale), block_overlap=(scale-1, scale-1) if overlap else (0,0), **res)])
      else:
        return FeatureExtractor(patch_size = (24,20), extractors = [bob.ip.LBP(8, radius=scale, **res)])
    else:
      if cls == 'MBLBP':
        return FeatureExtractor(patch_size = (24,20), template = bob.ip.LBP(8, block_size=(1,1), **res), block_overlap=overlap, square=square)
      else:
        return FeatureExtractor(patch_size = (24,20), template = bob.ip.LBP(8, radius=1, **res), square=square)

  else:
    if scale:
      if cls == 'MBLBP':
        res['lbp_extractors'] = [bob.ip.LBP(8, block_size=(scale,scale), overlap=(scale-1, scale-1) if overlap else (0,0))]
      else:
        res['lbp_extractors'] = [bob.ip.LBP(8, radius=scale, **res)]

    return LBP_CLASSES[cls](**res)


def command_line_options(command_line_arguments):

  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument('--databases', '-d', default=['banca'], nargs='+', help = "Select the databases to get the training images from.")

  parser.add_argument('--lbp-class', '-c', choices=LBP_CLASSES.keys(), default='MBLBP', help = "Specify, which type of LBP features are desired.")
  parser.add_argument('--lbp-variant', '-l', choices=LBP_VARIANTS.keys(), nargs='+', default = [], help = "Specify, which LBP variant(s) are wanted (ell is not available for MBLPB codes).")
  parser.add_argument('--lbp-overlap', '-o', action='store_true', help = "Specify the overlap of the MBLBP.")
  parser.add_argument('--lbp-cpp-implementation', '-I', action='store_true', help = "Use C++ or Python implementation of the feature extractor.")
  parser.add_argument('--lbp-scale', '-L', type=int, help="If given, only a single LBP extractor with the given LBP scale will be extracted, otherwise all possible scales are generated taken.")
  parser.add_argument('--lbp-square', '-Q', action='store_true', help="Generate only square feature extractors, and no rectangular ones.")

  parser.add_argument('--training-rounds', '-r', default=10, type=int, help = "The number of rounds to perform training during bootstrapping.")
  parser.add_argument('--bootstrapping-rounds', '-R', default=10, type=int, help = "The number of bootstrapping rounds to perform.")
  parser.add_argument('--patch-size', '-p', type=int, nargs=2, default=(24,20), help = "The size of the patch for the image in y and x.")
  parser.add_argument('--distance', '-s', type=int, default=2, help = "The distance with which the image should be scanned.")
  parser.add_argument('--scale-base', '-S', type=float, default=math.pow(2.,-1./4.), help = "The logarithmic distance between two scales (should be between 0 and 1).")
  parser.add_argument('--first-scale', '-f', type=float, default=math.sqrt(0.5), help = "The first scale of the image to consider (should be between 0 and 1, higher values will slow down the training process).")
  parser.add_argument('--similarity-thresholds', '-t', type=float, nargs=2, default=(0.3, 0.7), help = "The bounding box overlap thresholds for which negative (< thres[0]) and positive (> thers[1]) examples are accepted.")

  parser.add_argument('--examples-per-image-scale', '-e', type=int, nargs=2, default = [100, 100], help = "The number of positive and negative training examples for each image scale.")
  parser.add_argument('--training-examples', '-E', type=int, nargs=2, default = [10000, 10000], help = "The number of positive and negative training examples to sample.")
  parser.add_argument('--limit-training-files', '-y', type=int, help = "Limit the number of training files (for debug purposes only).")

  parser.add_argument('--trained-file', '-w', default = 'bootstrapped.hdf5', help = "The file to write the resulting trained detector into.")

  facereclib.utils.add_logger_command_line_option(parser)
  args = parser.parse_args(command_line_arguments)
  facereclib.utils.set_verbosity_level(args.verbose)

  return args



def main(command_line_arguments = None):
  args = command_line_options(command_line_arguments)

  # get training data
  training_files = utils.training_image_annot(args.databases, args.limit_training_files)

  # create the training set
  sampler = Sampler(patch_size=args.patch_size, scale_factor=args.scale_base, first_scale=args.first_scale, distance=args.distance, similarity_thresholds=args.similarity_thresholds, cpp_implementation=args.lbp_cpp_implementation)
  preprocessor = facereclib.preprocessing.NullPreprocessor()

  facereclib.utils.info("Loading %d training images" % len(training_files))
  i = 1
  all = len(training_files)
  for file_name, annotations in training_files:
    facereclib.utils.debug("Loading image file '%s' with %d faces" % (file_name, len(annotations)))
    sys.stdout.write("\rProcessing image file %d of %d '%s' " % (i, all, file_name))
    i += 1
    sys.stdout.flush()
    try:
      image = preprocessor(preprocessor.read_original_data(file_name))
      if args.lbp_cpp_implementation:
        boxes = [utils.bounding_box_from_annotation(**annotation) for annotation in annotations]
      else:
        boxes = [utils.BoundingBox(**annotation) for annotation in annotations]
      sampler.add(image, boxes, args.examples_per_image_scale[0], args.examples_per_image_scale[1])
    except KeyError as e:
      facereclib.utils.warn("Ignoring file '%s' since the eye annotations are incomplete" % file_name)
    except Exception as e:
      facereclib.utils.warn("Couldn't process file '%s': '%s'" % (file_name, e))
      raise
  sys.stdout.write("\n")


  # train the classifier using bootstrapping
  facereclib.utils.info("Extracting training features")
  feature_extractor = lbp_variant(args.lbp_class, args.lbp_variant, args.lbp_overlap, args.lbp_scale, args.lbp_square, args.lbp_cpp_implementation)
  trainer = xbob.boosting.core.boosting.Boost('LutTrainer', args.training_rounds, feature_extractor.maximum_label())
  bootstrapping = Bootstrap(number_of_rounds=args.bootstrapping_rounds, number_of_positive_examples_per_round=args.training_examples[0], number_of_negative_examples_per_round=args.training_examples[1])

  # perform the bootstrapping
  classifier = bootstrapping(trainer, sampler, feature_extractor, filename=args.trained_file)

  # write the machine and the feature extractor into the same HDF5 file
  save(args.trained_file, classifier, feature_extractor, args.lbp_cpp_implementation)
  facereclib.utils.info("Saved bootstrapped classifier to file '%s'" % args.trained_file)

