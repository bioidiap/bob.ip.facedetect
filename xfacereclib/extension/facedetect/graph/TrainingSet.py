import bob
import facereclib
import os
import numpy

from .._features import prune_detections, BoundingBox
from ..utils import best_detection, display

class TrainingSet:

  def __init__(self, sampler, cascade, patch_size = (96,80), offset = (20,10), training_image_directory = "training_set"):
    self.sampler = sampler
    self.cascade = cascade
    self.patch_size = patch_size
    self.offset = offset
    self.image_size = tuple([patch_size[i] + 2*offset[i] for i in (0,1)])
    self.training_image_directory = training_image_directory
    bob.io.create_directories_save(training_image_directory)

    self.fen = bob.ip.FaceEyesNorm(self.image_size[0], self.image_size[1], offset[0], offset[1], self.image_size[0] - offset[0], self.image_size[1] - offset[1])

    self.images = []
    self.annotations = []

  def add_image(self, image_file_name, annotations, image_file):
    """Detects the face in the image, if not done yet, and stores the image and the re-scaled annotations in the training directory."""
    if not annotations:
      return
    detected_image_file = image_file.make_path(self.training_image_directory, '.hdf5')
    detected_annotation_file = image_file.make_path(self.training_image_directory, '.pos')
    if os.path.exists(detected_image_file):
      # check if annotation file exists (if not, there was no image detected before)
      if os.path.exists(detected_annotation_file):
        # load detected image and annotations
        image = bob.io.load(detected_image_file)
        if image.shape == self.image_size:
          self.images.append(bob.io.load(detected_image_file))
          self.annotations.append(facereclib.utils.annotations.read_annotations(detected_annotation_file, 'named'))
        else:
          facereclib.utils.warn("Skipping image file '%s' since image resolution %s is not what we need: %s" % (image_file_name, str(image.shape), str(self.image_size)))
      else:
        facereclib.utils.warn("Skipping image file '%s' since face was not detected" % image_file_name)

    else:
      # detect the face in the image
      image = bob.ip.rgb_to_gray(bob.io.load(image_file_name))

      # limit the image such that only the face is visible (to speed up the processing)
      ys = [ann[0] for ann in annotations[0].itervalues()]
      xs = [ann[1] for ann in annotations[0].itervalues()]
      min_y, max_y, min_x, max_x = min(ys), max(ys), min(xs), max(xs)

      dy, dx = max_y - min_y, max_x - min_x
      top, bottom = max(min_y - 2*dy, 0), min(max_y + 2*dy, image.shape[0])
      left, right = max(min_x - 3*dx, 0), min(max_x + 3*dx, image.shape[1])
      image = image[top:bottom, left:right].copy()

      annotations = {key: (value[0]-top, value[1]-left) for key,value in annotations[0].iteritems()}

      detections, predictions = [], []
      for prediction, bounding_box in self.sampler.iterate_cascade(self.cascade, image):
        detections.append(bounding_box)
        predictions.append(prediction)
        facereclib.utils.debug("Found bounding box %s with value %f" % (str(bounding_box), prediction))

      # prune detections
      detections, predictions = prune_detections(detections, numpy.array(predictions), 1)
      bb, value = best_detection(detections, predictions, 0.2)

      # extract image with offset
      scaled_image = self.fen(image, bb.top_f, bb.left_f, bb.bottom_f, bb.right_f)
      # use the same transformation to normalize the annotations
      geon = self.fen.geom_norm
      geon.rotation_angle *= -1.
      eyes = self.fen.last_eye_center
      scaled_annotation = {key:geon(value, eyes[0], eyes[1]) for key,value in annotations.iteritems()}

      # write image and annotations
      bob.io.create_directories_save(os.path.dirname(detected_image_file))
      bob.io.save(scaled_image, detected_image_file)

      # check if scale annotations are inside the boundary (otherwise the face was not detected)
      for ann in scaled_annotation.itervalues():
        if ann[0] < 0 or ann[0] >= self.image_size[0] or ann[1] < 0 or ann[1] >= self.image_size[1]:
          facereclib.utils.warn("Could not detect face for image %s since annotation %s is outside the image size %s" % (image_file_name, str(ann), str(self.image_size)))
          return
      with open (detected_annotation_file, 'w') as a:
        a.write("\n".join(["%s %f %f" % (key, value[1], value[0]) for key, value in scaled_annotation.iteritems()]))

      display(scaled_image, scaled_annotation)

      self.images.append(scaled_image)
      self.annotations.append(scaled_annotation)

  def get_images_and_annotations(self, annotation_types):
    annotations = [{k:a[k] for k in annotation_types} for a in self.annotations]
    targets = []
    for annotation in annotations:
      target = numpy.ndarray((2*len(annotation_types),), numpy.float64)
      for i, key in enumerate(annotation_types):
        # we define the annotation positions relative to the center of the bounding box
        # This is different to Cosmins implementation, who used the top-left corner instead of the center
        target[2*i] = annotation[key][0]
        target[2*i+1] = annotation[key][1]
      targets.append(target)
    return self.images, targets
