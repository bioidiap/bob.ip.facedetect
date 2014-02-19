
import argparse
import facereclib
import bob
import numpy
import math
import xbob.boosting
import xbob.flandmark
import os
import matplotlib.pyplot

from .. import utils, detector
from .._features import prune_detections, FeatureExtractor

def command_line_options(command_line_arguments):

  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument('--database', '-d', default = 'banca', help = "Select the database to get the training images from.")
  parser.add_argument('--protocol', '-P', help = "If given, the test files from the given protocol are detected.")
  parser.add_argument('--limit-test-files', '-y', type=int, help = "Limit the test files to the given number (for debug purposes mainly)")
  parser.add_argument('--distance', '-s', type=int, default=2, help = "The distance with which the image should be scanned.")
  parser.add_argument('--scale-base', '-S', type=float, default = math.pow(2.,-1./16.), help = "The logarithmic distance between two scales (should be between 0 and 1).")
  parser.add_argument('--lowest-scale', '-f', type=float, default = 0.125, help = "Faces which will be lower than the given scale times the image resolution will not be found.")
  parser.add_argument('--cascade-file', '-r', required=True, help = "The file to read the cascade (or the strong classifier) from.")
  parser.add_argument('--prediction-threshold', '-T', type=float, help = "Detections with values below this threshold will be rejected by the detector.")
  parser.add_argument('--output-directory', '-o', help = "If given, the extracted faces will be written to the given directory.")
  parser.add_argument('--preprocessor', '-O', default='face-crop', help = "The preprocessor to be used to crop the faces.")
  parser.add_argument('--display', '-x', action='store_true', help = "If enabled, the detected faces will be displayed. Please start the program with 'ipython -pylab' to see the window.")
  parser.add_argument('--ground-truth-file', '-g', default="ground_truth.txt", help = "File to write the ground truth eyes.")
  parser.add_argument('--detection-error-file', '-e', default="detections.txt", help = "File to write the eyes located based on the detected faces.")
  parser.add_argument('--landmark-error-file', '-E', default="landmarks.txt", help = "File to write the landmarks of flandmark.")
  parser.add_argument('--best-detection-overlap', '-b', type=float, help = "If given, the average of the overlapping detections with this minimum overlap will be considered.")
  parser.add_argument('--include-landmarks', '-L', action='store_true', help = "Detect the landmarks in the face using flandmark?")
  parser.add_argument('--localizer-file', '-l', help = "Use the given localizer instead of flandmark.")

  facereclib.utils.add_logger_command_line_option(parser)
  args = parser.parse_args(command_line_arguments)
  facereclib.utils.set_verbosity_level(args.verbose)

  if args.localizer_file is not None:
    args.include_landmarks = True

  return args


def draw_box(image, bb, color):
  top = max(bb.top, 0)
  bottom = min(bb.bottom, image.shape[1]-1)
  left = max(bb.left, 0)
  right = min(bb.right, image.shape[2]-1)
  bob.ip.draw_box(image, y=top, x=left, height=bottom-top, width=right-left, color=color)

def main(command_line_arguments = None):
  args = command_line_options(command_line_arguments)

  # open database to collect test images
  test_files = utils.test_image_annot([args.database], [args.protocol], args.limit_test_files)
#  test_files = utils.training_image_annot([args.database], args.limit_test_files)

  facereclib.utils.info("Loading cascade from file %s" % args.cascade_file)
  hdf5 = bob.io.HDF5File(args.cascade_file)
  cascade = detector.Cascade(classifier_file=hdf5)

  # create the test examples
  preprocessor = facereclib.utils.resources.load_resource(args.preprocessor, 'preprocessor')
  sampler = detector.Sampler(distance=args.distance, scale_factor=args.scale_base, lowest_scale=args.lowest_scale)

  if args.include_landmarks:
    if args.localizer_file is not None:
      localizer, feature_extractor, _, _ = detector.load(args.localizer_file)
    else:
      flandmark = xbob.flandmark.Localizer()
    l = open(args.landmark_error_file, 'w')
    l.write("# --cascade-file %s --distance %d --scale-base %f --lowest-scale %s --prediction-threshold %s\n" % (args.cascade_file, args.distance, args.scale_base, args.lowest_scale, "None" if args.prediction_threshold is None else "%f" % args.prediction_threshold))
    l.write("# file y x y x ...\n")

  # iterate over the test files and detect the faces
  i = 1
  with open(args.detection_error_file, 'w') as e, open(args.ground_truth_file, 'w') as g:
    # write configuration
    e.write("# --cascade-file %s --distance %d --scale-base %f --lowest-scale %s --prediction-threshold %s\n" % (args.cascade_file, args.distance, args.scale_base, args.lowest_scale, "None" if args.prediction_threshold is None else "%f" % args.prediction_threshold))
    e.write("# file re-y re-x le-y le-x\n")
    g.write("# file re-y re-x le-y le-x\n")
    for filename, annotations, file in test_files:
      facereclib.utils.info("Loading image %d of %d from file '%s'" % (i, len(test_files), filename))
      i += 1
      image = preprocessor(preprocessor.read_original_data(filename))

      # get the detection scores for the image
      predictions = []
      detections = []
      for prediction, bounding_box in sampler.iterate_cascade(cascade, image):
        if args.prediction_threshold is None or prediction > args.prediction_threshold:
          predictions.append(prediction)
          detections.append(bounding_box)

      facereclib.utils.debug(".. number of detections: %d" % len(detections))

      # compute best location
      if args.best_detection_overlap is not None:
        # compute average over the best locations
        bb, value = utils.best_detection(detections, predictions, args.best_detection_overlap)
      else:
        # get the detection with the highest value
        d,p = prune_detections(detections, numpy.array(predictions), 0.01, 1)
        bb = d[0]

      # compute expected eye locations
      annots = utils.expected_eye_positions(bb)

      # get ground truth bounding boxes from annotations
      assert len(annotations) == 1
      gt = annotations[0]

      if args.include_landmarks:
        if args.localizer_file is not None:
          landmarks =  utils.localize(localizer, feature_extractor, image, bb)
        else:
          landmarks = utils.detect_landmarks(flandmark, image.astype(numpy.uint8), bb)
        l.write(("%s" + " %f %f"*len(landmarks) + "\n") % tuple([file.path] + [landmarks[a][b] for a in range(len(landmarks)) for b in range(2)]))


      g.write("%s %f %f %f %f\n" % (file.path, gt['reye'][0], gt['reye'][1], gt['leye'][0], gt['leye'][1]))
      e.write("%s %f %f %f %f\n" % (file.path, annots['reye'][0], annots['reye'][1], annots['leye'][0], annots['leye'][1]))

      if args.output_directory:
        output_filename = file.make_path(args.output_directory, '.png')
        cut = preprocessor(image, annots)
        bob.io.save(cut.astype(numpy.uint8), output_filename, True)
        facereclib.utils.info(".. wrote image %s" % output_filename)

      if args.display:
        colored = bob.ip.gray_to_rgb(image).astype(numpy.uint8)
        draw_box(colored, bb, color=(255,0,0))
        bob.ip.draw_cross(colored, y=int(annots['reye'][0]), x=int(annots['reye'][1]), radius=5, color=(255,0,0))
        bob.ip.draw_cross(colored, y=int(annots['leye'][0]), x=int(annots['leye'][1]), radius=5, color=(255,0,0))
        bb = utils.bounding_box_from_annotation(**gt)
        draw_box(colored, bb, color=(0,255,0))
        bob.ip.draw_cross_plus(colored, y=int(gt['reye'][0]), x=int(gt['reye'][1]), radius=5, color=(0,255,0))
        bob.ip.draw_cross_plus(colored, y=int(gt['leye'][0]), x=int(gt['leye'][1]), radius=5, color=(0,255,0))

        if args.include_landmarks:
          if len(landmarks):
            if args.localizer_file is None:
              lm = {
                'reye' : ((landmarks[1][0] + landmarks[5][0])/2., (landmarks[1][1] + landmarks[5][1])/2.),
                'leye' : ((landmarks[2][0] + landmarks[6][0])/2., (landmarks[2][1] + landmarks[6][1])/2.)
              }
            else:
              lm = {
                'reye' : landmarks[0],
                'leye' : landmarks[1]
              }
            bb = utils.bounding_box_from_annotation(**lm)
            draw_box(colored, bb, color=(0,0,255))
            bob.ip.draw_box(colored, y=int(lm['reye'][0]) - 5, x=int(lm['reye'][1]) - 5, height = 10, width=10,  color=(0,0,255))
            bob.ip.draw_box(colored, y=int(lm['leye'][0]) - 5, x=int(lm['leye'][1]) - 5, height = 10, width=10,  color=(0,0,255))

        matplotlib.pyplot.clf()
        matplotlib.pyplot.imshow(numpy.rollaxis(numpy.rollaxis(colored, 2),2))
        matplotlib.pyplot.draw()


