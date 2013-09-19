
import argparse
import facereclib
import bob
import numpy
import math
import xbob.boosting
import os

from ..utils import  BoundingBox
from ..detector import Sampler, MBLBPFeatures, load_features

def command_line_options(command_line_arguments):

  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument('--database', '-d', default = 'banca', help = "Select the database to get the training images from.")
  parser.add_argument('--distance', '-s', type=int, default=5, help = "The distance with which the image should be scanned.")
  parser.add_argument('--scale-base', '-S', type=float, default = math.pow(2.,-1./4.), help = "The logarithmic distance between two scales (should be between 0 and 1).")
  parser.add_argument('--first-scale', '-f', type=float, default = 0.5, help = "The first scale of the image to consider (should be between 0 and 1, higher values will slow down the detection process).")
  parser.add_argument('--trained-file', '-r', default = 'detector.hdf5', help = "The file to write the resulting trained detector into.")
  parser.add_argument('--write-best', '-w', help = "Writes the best detection for each file into the given directory")
  parser.add_argument('--prediction-threshold', '-t', default = 0., type = float, help = "The threshold for which a detection should be classified as positive.")
  parser.add_argument('--limit-probe-files', '-y', type=int, help = "Limit the number of test files (for debug purposes only).")

  facereclib.utils.add_logger_command_line_option(parser)
  args = parser.parse_args(command_line_arguments)
  facereclib.utils.set_verbosity_level(args.verbose)

  return args



def main(command_line_arguments = None):
  args = command_line_options(command_line_arguments)

  # open database to collect training images
  database = facereclib.utils.resources.load_resource(args.database, 'database')
  probe_files = database.probe_files()
  probe_files = [probe_files[t] for t in facereclib.utils.quasi_random_indices(len(probe_files), args.limit_probe_files)]

  # load classifier and feature extractor
  f = bob.io.HDF5File(args.trained_file)
  f.cd("/Machine")
  classifier = xbob.boosting.core.boosting.BoostMachine(hdf5file=f)
  f.cd("/Features")
  feature_extractor = load_features(f)
  feature_extractor.set_model(classifier)

  # create the test examples
  preprocessor = facereclib.preprocessing.NullPreprocessor()
  sampler = Sampler(distance=args.distance, scale_factor=args.scale_base, first_scale=args.first_scale)

  # iterate over the probe files and detect the faces
  for file in probe_files:
    filename = str(file.make_path(database.original_directory, database.original_extension))
    facereclib.utils.info("Loading image file '%s'" % filename)
    image = preprocessor(preprocessor.read_original_data(filename))

    # get the detection scores for the image
    bounding_boxes = []
    predictions = []
    for bounding_box, features in sampler.iterate(image, feature_extractor):
      prediction = classifier(features)
      if prediction > args.prediction_threshold:
        predictions.append(prediction)
        bounding_boxes.append(bounding_box)
        facereclib.utils.debug("Found bounding box %s with value %f" % (str(bounding_box), prediction))


    if args.write_best:
      if len(predictions) == 0:
        facereclib.utils.warn("There was no bounding box found for image %s with value > %f" % (filename, args.prediction_threshold))
      else:
        filename = str(file.make_path(args.write_best, '.png'))
        facereclib.utils.ensure_dir(os.path.dirname(filename))

        max_index = numpy.argmax(predictions)
        best_bb = bounding_boxes[max_index]

        facereclib.utils.info("Writing extracted face to file '%s', with box '%s' and prediction '%f'" % (filename, str(best_bb), predictions[max_index]))
        bob.io.save(best_bb.extract(image).astype(numpy.uint8), filename)

#    positives = [(bb, pred) for bb, pred in zip(bounding_boxes, predictions) if pred > 0]

