"""Collects training data including ground truth bounding boxes and writes them into file lists."""

import argparse
import bob.ip.facedetect
import bob.core
import importlib
import os

logger = bob.core.log.setup('bob.ip.facedetect')


def command_line_options(command_line_arguments):

  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument('--database', '-d', help = "Select the database to get the training images from; only the world group is used.")
  parser.add_argument('--protocols', '-p', nargs='+', help = "Select the protocols for the database (if necessary).")
  parser.add_argument('--groups', '-g', nargs='+', default=["world"], help = "Select the groups of images to be used.")
  parser.add_argument('--image-directory', '-i', required = True, help = "Select the image directory to load the training images from.")
  parser.add_argument('--image-extensions', '-e', nargs='+', help = "The filename extensions of the images (if not already given by the --database).")
  parser.add_argument('--annotation-directory', '-a', help = "Select the annotation directory to get the annotation files from.")
  parser.add_argument('--annotation-extension', '-E', default = '.pos', help = "Select the filename extension of the annotation files.")
  parser.add_argument('--annotation-type', '-t', default = 'named', help = "Select the type the the annotation files (i.e., how to read them).")
  parser.add_argument('--no-annotations', '-n', action = 'store_true', help = "If specified, annotations for these files are not necessary since the files do not contain faces.")

  parser.add_argument('--output-file', '-w', required=True, help = "Select the file list containing training images to be written")
  parser.add_argument('--force', '-F', action='store_true', help = "Force the re-creation of the --output-file")

  bob.core.log.add_command_line_option(parser)
  args = parser.parse_args(command_line_arguments)
  bob.core.log.set_verbosity_level(logger, args.verbose)


  # sanity checks
  if not os.path.exists(args.image_directory):
    raise ValueError("The given image directory '%s' does not exist" % args.image_directory)

  if args.annotation_directory is not None and not os.path.exists(args.annotation_directory):
    raise ValueError("The given annotation directory '%s' does not exist" % args.annotation_directory)

  if args.database is None and args.annotation_directory is None and not args.no_annotations:
    raise ValueError("When scanning for images, please specify the --annotation-directory, or use the --no-annotations flag to disable searching for annotations")

  return args



def main(command_line_arguments = None):
  args = command_line_options(command_line_arguments)

  if os.path.exists(args.output_file) and not args.force:
    logger.info("The given output file '%s' already exists. Use --force to force re-creation.", args.output_file)
    return

  train_set = bob.ip.facedetect.train.TrainingSet(feature_directory = None)

  # check if a database is given
  if args.database is not None:
    # collect arguments of the database interface
    logger.info("Loading data of database bob.db.%s", args.database)
    kwargs = dict(original_directory=args.image_directory)
    if args.image_extensions is not None:
      kwargs['original_extension'] = args.image_extensions[0]
    if args.annotation_directory is not None:
      kwargs['annotation_directory'] = args.annotation_directory

    # import database
    db = importlib.import_module("bob.db." + args.database).Database(**kwargs)
    # collect File objects
    files = db.all_files(groups=args.groups) if args.protocols is None else db.uniquify([f for p in args.protocols for f in db.all_files(groups=args.groups, protocol=p)])
    # add files
    train_set.add_from_db(db, files)

  else:
    # collect data by searching for all images in the given image directory
    logger.info("Collecting training images from directory %s", args.image_directory)

    for directory, _, files in os.walk(args.image_directory):
      for f in files:
        base, ext = os.path.splitext(f)
        if ext in args.image_extensions:
          # check if annotation file exists
          if args.no_annotations:
            train_set.add_image(os.path.join(directory, f), [])
          else:
            annot = os.path.join(args.annotation_directory, os.path.relpath(directory, args.image_directory), base + args.annotation_extension)
            if os.path.exists(annot):
              annotations = bob.ip.facedetect.train.read_annotation_file(annot, args.annotation_type)
              train_set.add_image(os.path.join(directory, f), annotations)
            else:
              logger.debug("Couldn't find annotation file '%s' for image file '%s'", annot, os.path.join(directory, f))


  logger.info("Saving training set to file %s", args.output_file)
  train_set.save(args.output_file)
