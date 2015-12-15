"""Extracts features for the training set of the given file lists using the given feature extractor."""

import argparse
import bob.ip.facedetect
import importlib
import os, math

import bob.core
logger = bob.core.log.setup('bob.ip.facedetect')



# create feature extractor
LBP_VARIANTS = {
  'ell'  : {'circular' : True},
  'u2'   : {'uniform' : True},
  'ri'   : {'rotation_invariant' : True},
  'mct'  : {'to_average' : True, 'add_average_bit' : True},
  'tran' : {'elbp_type' : 'transitional'},
  'dir'  : {'elbp_type' : 'direction-coded'}
}


def command_line_options(command_line_arguments):

  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument('--file-lists', '-i', nargs='+', help = "Select the training lists to extract features for.")
  parser.add_argument('--feature-directory', '-d', default = "features", help = "The output directory, where features will be stores")
  parser.add_argument('--parallel', '-P', type=int, help = "Use this option to run the script in parallel in the SGE grid, using the given number of parallel processes")

  parser.add_argument('--patch-size', '-p', type=int, nargs=2, default=(24,20), help = "The size of the patch for the image in y and x.")
  parser.add_argument('--distance', '-s', type=int, default=2, help = "The distance with which the image should be scanned.")
  parser.add_argument('--scale-base', '-S', type=float, default=math.pow(2.,-1./8.), help = "The logarithmic distance between two scales (should be between 0 and 1).")
  parser.add_argument('--negative-examples-every', '-N', type=int, default=4, help = "Use only every nth scale to extract negative examples.")
  parser.add_argument('--lowest-scale', '-f', type=float, default=0, help = "Patches which will be lower than the given scale times the image resolution will not be taken into account; if 0. (the default) all patches will be considered.")
  parser.add_argument('--similarity-thresholds', '-t', type=float, nargs=2, default=(0.2, 0.8), help = "The bounding box overlap thresholds for which negative (< thres[0]) and positive (> thers[1]) examples are accepted.")
  parser.add_argument('--no-mirror-samples', '-M', action='store_true', help = "Disable mirroring of the training samples.")
  parser.add_argument('--examples-per-image-scale', '-e', type=int, nargs=2, default = [100, 100], help = "The number of positive and negative training examples for each image scale.")

  parser.add_argument('--lbp-multi-block', '-m', action='store_true', help = "If given multi-block LBP features will be extracted (otherwise, it's regular LBP).")
  parser.add_argument('--lbp-variant', '-l', choices=LBP_VARIANTS.keys(), nargs='+', default = [], help = "Specify, which LBP variant(s) are wanted (ell is not available for MBLPB codes).")
  parser.add_argument('--lbp-overlap', '-o', action='store_true', help = "Specify the overlap of the MBLBP.")
  parser.add_argument('--lbp-scale', '-L', type=int, help="If given, only a single LBP extractor with the given LBP scale will be extracted, otherwise all possible scales are generated.")
  parser.add_argument('--lbp-square', '-Q', action='store_true', help="Generate only square feature extractors, and no rectangular ones.")

  bob.core.log.add_command_line_option(parser)
  args = parser.parse_args(command_line_arguments)
  bob.core.log.set_verbosity_level(logger, args.verbose)

  return args



def main(command_line_arguments = None):
  args = command_line_options(command_line_arguments)

  train_set = bob.ip.facedetect.train.TrainingSet(feature_directory = args.feature_directory)

  # create feature extractor
  res = {}
  for t in args.lbp_variant:
    res.update(LBP_VARIANTS[t])
  if args.lbp_scale is not None:
    if args.lbp_multi_block:
      feature_extractor = bob.ip.facedetect.FeatureExtractor(patch_size = args.patch_size, extractors = [bob.ip.base.LBP(8, block_size=(args.lbp_scale,args.lbp_scale), block_overlap=(args.lbp_cale-1, args.lbp_scale-1) if args.lbp_overlap else (0,0), **res)])
    else:
      feature_extractor = bob.ip.facedetect.FeatureExtractor(patch_size = args.patch_size, extractors = [bob.ip.base.LBP(8, radius=args.lbp_scale, **res)])
  else:
    if args.lbp_multi_block:
      feature_extractor = bob.ip.facedetect.FeatureExtractor(patch_size = args.patch_size, template = bob.ip.base.LBP(8, block_size=(1,1), **res), overlap=args.lbp_overlap, square=args.lbp_square)
    else:
      feature_extractor = bob.ip.facedetect.FeatureExtractor(patch_size = args.patch_size, template = bob.ip.base.LBP(8, radius=1, **res), square=args.lbp_square)

  # load training sets
  for file_list in args.file_lists:
    logger.info("Loading file list %s", file_list)
    train_set.load(file_list)

  # generate sampler
  sampler = bob.ip.facedetect.detector.Sampler(patch_size=args.patch_size, scale_factor=args.scale_base, lowest_scale=args.lowest_scale, distance=args.distance)

  # extract features
  train_set.extract(sampler, feature_extractor, number_of_examples_per_scale = args.examples_per_image_scale, similarity_thresholds = args.similarity_thresholds, parallel = args.parallel, mirror = not args.no_mirror_samples, use_every_nth_negative_scale = args.negative_examples_every)
