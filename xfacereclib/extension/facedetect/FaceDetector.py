import facereclib

import math
import bob
import numpy

try:
  import xfacereclib.extension.facedetect
except:
  pass

class FaceDetector (facereclib.preprocessing.Preprocessor):

  def __init__(
        self,
        cascade,
        cropped_image_size,
        detection_threshold = 0,
        detection_overlap = 0.2,
        distance = 2,
        scale_base = math.pow(2., -1./16.),
        lowest_scale = 0.5,
        post_processor = "face-crop",
        color_channel = 'gray',
        **kwargs
  ):
    """Performs a face detection in the given image (ignoring any annotations)."""
    # call base class constructor
    facereclib.preprocessing.Preprocessor.__init__(
        self,
        cascade = cascade,
        cropped_image_size = cropped_image_size,
        detection_threshold = detection_threshold,
        detection_overlap = detection_overlap,
        distance = distance,
        scale_base = scale_base,
        lowest_scale = lowest_scale,
        post_processor = post_processor,
        color_channel = color_channel,
        **kwargs
    )

    self.m_sampler = xfacereclib.extension.facedetect.Sampler(scale_factor=scale_base, first_scale=lowest_scale, distance=distance)
    self.m_cascade = xfacereclib.extension.facedetect.Cascade(classifier_file=bob.io.HDF5File(cascade))
    self.m_detection_threshold = detection_threshold
    self.m_detection_overlap = detection_overlap
    self.m_color_channel = color_channel
    self.m_last_prediction = None
    self.m_post_processor = facereclib.utils.resources.load_resource(post_processor, "preprocessor")
    # overwrite the cropped positions of the post processor to use the top-left and bottom-right bounding box values
#    self.m_post_processor.m_cropped_positions = {'topleft':(0,0), 'bottomright':(cropped_image_size[0]-1, cropped_image_size[1]-1)}


  def __call__(self, image, annotation):
    # convert to the desired color channel
    image = facereclib.utils.gray_channel(image, self.m_color_channel)

    # detect the face
    detections = []
    predictions = []
    for prediction, bounding_box in self.m_sampler.iterate_cascade(self.m_cascade, image):
      if prediction > self.m_detection_threshold:
        detections.append(bounding_box)
        predictions.append(prediction)

    if not detections:
      utils.warning("No face found")
      return None

    bb, value = xfacereclib.extension.facedetect.utils.best_detection(detections, predictions, self.m_detection_overlap)
    annots = xfacereclib.extension.facedetect.utils.expected_eye_positions(bb)

    self.m_last_prediction = value

    # perform preprocessing
    return self.m_post_processor(image, annotations=annots)


  def quality(self):
    """Returns the quality of the last detection step."""
    return self.m_last_prediction

