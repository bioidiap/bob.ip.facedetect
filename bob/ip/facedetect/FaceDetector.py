import facereclib

import math
import numpy

try:
  import xfacereclib.extension.facedetect
  import bob.ip.flandmark
except ImportError as e:
  facereclib.utils.error("Import Error: %s" % e)

import bob.io.base

class FaceDetector (facereclib.preprocessing.Preprocessor):

  def __init__(
        self,
        cascade,
        detection_threshold = 0,
        detection_overlap = 0.2,
        distance = 2,
        scale_base = math.pow(2., -1./16.),
        lowest_scale = 0.125,
        post_processor = "face-crop",
        use_flandmark = False,
        color_channel = 'gray',
        **kwargs
  ):
    """Performs a face detection in the given image (ignoring any annotations)."""
    # call base class constructor
    facereclib.preprocessing.Preprocessor.__init__(
        self,
        cascade = cascade,
        detection_threshold = detection_threshold,
        detection_overlap = detection_overlap,
        distance = distance,
        scale_base = scale_base,
        lowest_scale = lowest_scale,
        post_processor = post_processor,
        color_channel = color_channel,
        **kwargs
    )

    self.m_sampler = xfacereclib.extension.facedetect.Sampler(scale_factor=scale_base, lowest_scale=lowest_scale, distance=distance)
    self.m_cascade = xfacereclib.extension.facedetect.Cascade(classifier_file=bob.io.base.HDF5File(cascade))
    self.m_detection_threshold = detection_threshold
    self.m_detection_overlap = detection_overlap
    self.m_color_channel = color_channel
    self.m_last_prediction = None
    if isinstance(post_processor, facereclib.preprocessing.Preprocessor):
      self.m_post_processor = post_processor
    else:
      self.m_post_processor = facereclib.utils.resources.load_resource(post_processor, "preprocessor")
    self.m_flandmark = bob.ip.flandmark.Flandmark() if use_flandmark else None

    # overwrite the cropped positions of the post processor to use the top-left and bottom-right bounding box values
#    self.m_post_processor.m_cropped_positions = {'topleft':(0,0), 'bottomright':(cropped_image_size[0], cropped_image_size[1])}

  def _landmarks(self, image, bounding_box):
    # get the landmarks in the face
    if self.m_flandmark is not None:
      # use the flandmark detector
      landmarks = xfacereclib.extension.facedetect.utils.detect_landmarks(self.m_flandmark, image, bounding_box)
      if landmarks is not None and len(landmarks):
        annots = {
          'reye' : ((landmarks[1][0] + landmarks[5][0])/2., (landmarks[1][1] + landmarks[5][1])/2.),
          'leye' : ((landmarks[2][0] + landmarks[6][0])/2., (landmarks[2][1] + landmarks[6][1])/2.)
        }
      else:
        facereclib.utils.warn("Could not detect landmarks -- using default landmarks")
        annots = xfacereclib.extension.facedetect.utils.expected_eye_positions(bounding_box)

    else:
      # estimate from default locations
      annots = xfacereclib.extension.facedetect.utils.expected_eye_positions(bounding_box)

    return annots

  def __call__(self, image, annotation):
    # convert to the desired color channel
    image = facereclib.utils.gray_channel(image, self.m_color_channel)

    # if annotations are given, use these annotations to initialize the bounding box, instead of detecting the face
    if annotation is not None and self.m_flandmark is not None and 'leye' in annotation and 'reye' in annotation:
      bounding_box = xfacereclib.extension.facedetect.utils.bounding_box_from_annotation(source='eyes', **annotation)
      self.m_annots = self._landmarks(image, bounding_box)
      detected = True
      for lm in ('reye', 'leye'):
        for i in range(2):
          detected = detected and (self.m_annots[lm][i] - annotation[lm][i] < 5)
      # if the annotations are close enough to the old annotations, we don't need to perform face detection
      # NOTE: the quality is NOT UPDATED in this case!
      if detected:
        facereclib.utils.debug("Skipping face detection since new annotations %s are close enough to last annotations %s" % (str(self.m_annots), str(annotation)))
        return self.m_post_processor(image, annotations=self.m_annots)
      facereclib.utils.debug("Running face detection since new annotations %s are to far from last annotations %s" % (str(self.m_annots), str(annotation)))


    # detect the face
    detections = []
    predictions = []
    for prediction, bounding_box in self.m_sampler.iterate_cascade(self.m_cascade, image):
      if prediction > self.m_detection_threshold:
        detections.append(bounding_box)
        predictions.append(prediction)

    if not detections:
      facereclib.utils.warn("No face found")
      return None

    bb, self.m_last_prediction = xfacereclib.extension.facedetect.utils.best_detection(detections, predictions, self.m_detection_overlap)

    self.m_annots = self._landmarks(image, bb)

    # perform preprocessing
    return self.m_post_processor(image, annotations=self.m_annots)


  def quality(self):
    """Returns the quality of the last detection step."""
    return self.m_last_prediction

