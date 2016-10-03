
import argparse
import numpy
import math
import os

import pkg_resources

import bob.io.base
import bob.io.image
import bob.ip.color
import bob.ip.facedetect
import bob.core
logger = bob.core.log.setup("bob.ip.facedetect")


def command_line_options(command_line_arguments):

  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument('--file-lists', '-i', nargs='+', default = [], help = "Select the file lists including to ground truth bounding boxes to evaluate.")
  parser.add_argument('--limit-test-files', '-y', type=int, help = "Limit the test files to the given number (for debug purposes mainly)")
  parser.add_argument('--distance', '-s', type=int, default=2, help = "The distance with which the image should be scanned.")
  parser.add_argument('--scale-base', '-S', type=float, default = math.pow(2.,-1./8.), help = "The logarithmic distance between two scales (should be between 0 and 1).")
  parser.add_argument('--lowest-scale', '-f', type=float, default = 0.0625, help = "Faces which will be lower than the given scale times the image resolution will not be found.")
  parser.add_argument('--cascade-file', '-r', default = pkg_resources.resource_filename("bob.ip.facedetect", "MCT_cascade.hdf5"), help = "The file to read the cascade from (has a proper default).")
  parser.add_argument('--prediction-threshold', '-T', type=float, help = "Detections with values below this threshold will be rejected by the detector.")
  parser.add_argument('--score-file', '-w', default='cascaded_scores.txt', help = "The score file to be written.")
  parser.add_argument('--prune-detections', '-p', type=float, default = 0.2, help = "If given, detections that overlap with the given threshold are pruned")
  parser.add_argument('--detection-threshold', '-j', type=float, default=0.5, help = "The overlap from Ground Truth for which a detection should be considered as successful")

  bob.core.log.add_command_line_option(parser)
  args = parser.parse_args(command_line_arguments)
  bob.core.log.set_verbosity_level(logger, args.verbose)

  return args


def main(command_line_arguments = None):
  args = command_line_options(command_line_arguments)

  # load cascade
  logger.info("Loading cascade from file %s", args.cascade_file)
  cascade = bob.ip.facedetect.detector.Cascade(bob.io.base.HDF5File(args.cascade_file))

  # collect test images
  train_set = bob.ip.facedetect.train.TrainingSet()
  for file_list in args.file_lists:
    logger.info("Loading file list %s", file_list)
    train_set.load(file_list)

  # create the test examples
  sampler = bob.ip.facedetect.detector.Sampler(patch_size=cascade.extractor.patch_size, distance=args.distance, scale_factor=args.scale_base, lowest_scale=args.lowest_scale)

  # iterate over the test files and detect the faces
  i = 1
  with open(args.score_file, 'w') as f:
    # write configuration
    f.write("# --cascade-file %s --distance %d --scale-base %f --lowest-scale %s --prediction-threshold %s --detection-threshold %f\n" % (args.cascade_file, args.distance, args.scale_base, args.lowest_scale, str(args.prediction_threshold) if args.prediction_threshold is not None else "None", args.detection_threshold))
    for image, ground_truth, file_name in train_set.iterate(args.limit_test_files):
      logger.info("Loading image %d of %d from %s", i, args.limit_test_files or len(train_set), file_name)

      # get the detection scores for the image
      predictions = []
      detections = []
      for prediction, bounding_box in sampler.iterate_cascade(cascade, image, args.prediction_threshold):
        predictions.append(prediction)
        detections.append(bounding_box)

      logger.info("Number of detections: %d", len(detections))

      # prune detections
      detections, predictions = bob.ip.facedetect.prune_detections(detections, numpy.array(predictions), args.prune_detections)
      logger.info("Number of pruned detections: %d", len(detections))

      # get ground truth bounding boxes from annotations
      f.write("%s %d\n" % (file_name, len(ground_truth)))

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
      i += 1
