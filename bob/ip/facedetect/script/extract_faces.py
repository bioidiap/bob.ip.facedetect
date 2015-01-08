"""A script to extract faces from a given image or database using the given cascade.

If wanted, the resulting faces can be split into several files and written to the given output directory."""

import argparse
import facereclib
import numpy
import math
import os
import pkg_resources

import xfacereclib.extension.facedetect

import bob.io.base
import bob.io.image

from .. import utils, detector
from .._features import prune_detections, FeatureExtractor

from matplotlib import pyplot, patches

def command_line_options(command_line_arguments):

  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument('--image', '-i', help = "Select the image to detect the face in.")
  parser.add_argument('--database', '-d', help = "Select the database to read images from.")
  parser.add_argument('--display', '-x', action='store_true', help = "Display the detected images.")

  parser.add_argument('--distance', '-s', type=int, default=2, help = "The distance with which the image should be scanned.")
  parser.add_argument('--scale-base', '-S', type=float, default = math.pow(2.,-1./16.), help = "The logarithmic distance between two scales (should be between 0 and 1).")
  parser.add_argument('--lowest-scale', '-f', type=float, default = 0.125, help = "Faces which will be lower than the given scale times the image resolution will not be found.")
  parser.add_argument('--cascade-file', '-r', help = "The file to read the cascade from (has a proper default).")

  parser.add_argument('--prediction-threshold', '-T', type=float, default = 25, help = "Detections with values below this threshold will be rejected by the detector.")
  parser.add_argument('--prune-detections', '-p', type=float, default = 0.1, help = "If given, detections that overlap with the given threshold are pruned")
  parser.add_argument('--output-directories', '-O', nargs="+", default = None, help = "If given, detected faces (limited to 100 per image) will be extracted and written to the given output directory; must be the same count as '--output-processors'.")
  parser.add_argument('--output-processors', '-o', nargs="+", default = ["face-crop"], help = "The list of Preprocessor classes (from the facereclib) to apply on the detected faces to write them to file")

  facereclib.utils.add_logger_command_line_option(parser)
  args = parser.parse_args(command_line_arguments)
  facereclib.utils.set_verbosity_level(args.verbose)

  if args.cascade_file is None:
    args.cascade_file = pkg_resources.resource_filename('xfacereclib.extension.facedetect', 'MCT_cascade.hdf5')

  return args


def main(command_line_arguments = None):
  args = command_line_options(command_line_arguments)

  if args.database is not None:
    files = xfacereclib.extension.facedetect.utils.test_image_annot([args.database], [None], limit=None)
    images = [f[0] for f in files]
  elif args.image is not None:
    images = [args.image]
  else:
    facereclib.utils.error("Specify either a database or an image file")

  # load cascade
  facereclib.utils.info("Loading cascade from file %s" % args.cascade_file)
  cascade = detector.Cascade(classifier_file=bob.io.base.HDF5File(args.cascade_file))

  # create the test examples
  postprocessors = [facereclib.utils.resources.load_resource(args.output_processors[i], "preprocessor") for i in range(len(args.output_processors))]
  sampler = detector.Sampler(distance=args.distance, scale_factor=args.scale_base, lowest_scale=args.lowest_scale)

  # iterate over the test files and detect the faces
  count = 1
  for filename in images:
    facereclib.utils.info("Loading image %d of %d from file '%s'" % (count, len(images), filename))
    count += 1
    try:
      image = postprocessors[0](postprocessors[0].read_original_data(filename))
    except Exception as e:
      facereclib.utils.error("Skipping file %s since: %s" % (filename, e))

    # get the detection scores for the image
    predictions = []
    detections = []
    for prediction, bounding_box in sampler.iterate_cascade(cascade, image):
      predictions.append(prediction)
      detections.append(bounding_box)

    # prune detections
    pruned, predictions = prune_detections(detections, numpy.array(predictions), args.prune_detections)
    thresholded = [pruned[0]] + [pruned[i] for i in range(1, len(predictions)) if predictions[i] > args.prediction_threshold]
    facereclib.utils.info("Number of pruned detections over threshold: %d (of %d) detections" % (len(thresholded), len(pruned)))

    if args.display:
      pyplot.clf()
      pyplot.imshow(image, cmap='gray')
      for i, bb in enumerate(thresholded):
        v = max((predictions[i] - args.prediction_threshold) / (predictions[0] - args.prediction_threshold) * 0.5 + 0.5, 0.5)
        color = (v,0,0)
        r = patches.Rectangle((bb.topleft[1], bb.topleft[0]), bb.size_f[1], bb.size_f[0], facecolor='none', edgecolor=color, linewidth=2.5)
        pyplot.gca().add_patch(r)
      pyplot.draw()

    if args.output_directories is not None:
      for proc, direct in zip(postprocessors,args.output_directories):
        fname = os.path.join(direct, os.path.splitext(os.path.basename(filename))[0] + "_%02d.png")
        facereclib.utils.info("Writing %d images to %s" % (min(100, len(thresholded)), fname))
        for i in range(min(100, len(thresholded))):
          cropped_filename = fname % i
          # estimate eye positions based on extracted image
          annot = utils.expected_eye_positions(thresholded[i])
          cropped_image = proc(image, annot).astype(numpy.uint8)
          bob.io.base.save(cropped_image, cropped_filename, True)

  raw_input("Press Enter to finish")

