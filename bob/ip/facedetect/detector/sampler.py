
import math
from .._library import BoundingBox

import bob.ip.base


class Sampler:
  """This class generates (samples) bounding boxes for different scales and locations in the image.

  It computes different scales of the image and provides a tight set of :py:class:`BoundingBox` of a given patch size for the given image.

  **Constructor Documentation:**

    Generates a patch-sampler, which will scan images and sample bounding boxes.

    **Parameters:**

    ``patch_size`` : (int, int)
      the size of the patch (i.e., the bounding box) to sample

    ``scale_factor`` : float
      image pyramids are computed using the given scale factor between two scales

    ``lowest_scale`` : float or None
      patches which will be lower than the given scale times the image resolution will not be taken into account;
      if 0. all possible patches will be considered

    ``distance`` : int
      the distance in both horizontal and vertical direction to generate samples
  """

  def __init__(self, patch_size = (24,20), scale_factor = math.pow(2., -1./16.), lowest_scale = math.pow(2., -6.), distance = 2):

    self.m_patch_box = BoundingBox((0, 0), patch_size)
    self.m_scale_factor = scale_factor
    self.m_lowest_scale = lowest_scale
    self.m_distance = distance


  def scales(self, image):
    """scales(image) -> scale, shape

    Computes the all possible scales for the given image and yields a tuple of the scale and the scaled image shape as an iterator.

    **Parameters::**

    ``image`` : array_like(2D or 3D)
      The image, for which the scales should be computed

    **Yields:**

    ``scale`` : float
      The next scale of the image to be considered

    ``shape`` : (int, int) or (int, int, int)
      The shape of the image, when scaled with the current ``scale``
    """
    # compute the minimum scale so that the patch size still fits into the given image
    minimum_scale = max(self.m_patch_box.size_f[0] / image.shape[-2], self.m_patch_box.size_f[1] / image.shape[-1])
    if self.m_lowest_scale:
      maximum_scale = min(minimum_scale / self.m_lowest_scale, 1.)
    else:
      maximum_scale = 1.
    current_scale_power = 0.

    # iterate over all possible scales
    while True:
      # scale the image
      scale = minimum_scale * math.pow(self.m_scale_factor, current_scale_power)
      if scale > maximum_scale:
        # image is smaller than the requested minimum size
        break
      current_scale_power -= 1.
      scaled_image_shape = bob.ip.base.scaled_output_shape(image, scale)

      # return both the scale and the scaled image size
      yield scale, scaled_image_shape


  def sample_scaled(self, shape):
    """sample_scaled(shape) -> bounding_box

    Yields an iterator that iterates over all sampled bounding boxes in the given (scaled) image shape.

    **Parameters:**

    ``shape`` : (int, int) or (int, int, int)
      The (current) shape of the (scaled) image

    **Yields:**

    ``bounding_box`` : :py:class:`BoundingBox`
      An iterator iterating over all bounding boxes that are valid for the given shape
    """
    for y in range(0, shape[-2]-self.m_patch_box.bottomright[0], self.m_distance):
      for x in range(0, shape[-1]-self.m_patch_box.bottomright[1], self.m_distance):
        # create bounding box for the current shift
        yield self.m_patch_box.shift((y,x))


  def sample(self, image):
    """sample(image) -> bounding_box

    Yields an iterator over all bounding boxes in different scales that are sampled for the given image.

    **Parameters:**

    ``image`` : array_like(2D or 3D)
      The image, for which the bounding boxes should be generated

    **Yields:**

    ``bounding_box`` : :py:class:`BoundingBox`
      An iterator iterating over all bounding boxes for the given ``image``
    """
    for scale, scaled_image_shape in self.scales(image):
      # prepare the feature extractor to extract features from the given image
      for bb in self.sample_scaled(scaled_image_shape):
        # extract features for
        yield bb.scale(1./scale)


  def iterate(self, image, feature_extractor, feature_vector):
    """iterate(image, feature_extractor, feature_vector) -> bounding_box

    Scales the given image, and extracts features from all possible bounding boxes.

    For each of the sampled bounding boxes, this function fills the given pre-allocated feature vector and yields the current bounding box.

    **Parameters:**

    ``image`` : array_like(2D)
      The given image to extract features for

    ``feature_extractor`` : :py:class:`FeatureExtractor`
      The feature extractor to use to extract the features for the sampled patches

    ``feature_vector`` : :py:class:`numpy.ndarray` (1D, uint16)
      The pre-allocated feature vector that will be filled inside this function; needs to be of size :py:attr:`FeatureExtractor.number_of_features`

    **Yields:**

    ``bounding_box`` : :py:class:`BoundingBox`
      The bounding box for which the current features are extracted for
    """
    for scale, scaled_image_shape in self.scales(image):
      # prepare the feature extractor to extract features from the given image
      feature_extractor.prepare(image, scale)
      for bb in self.sample_scaled(scaled_image_shape):
        # extract features for
        feature_extractor.extract_indexed(bb, feature_vector)
        yield bb.scale(1./scale)


  def iterate_cascade(self, cascade, image, threshold = None):
    """iterate_cascade(self, cascade, image, [threshold]) -> prediction, bounding_box

    Iterates over the given image and computes the cascade of classifiers.
    This function will compute the cascaded classification result for the given ``image`` using the given ``cascade``.
    It yields a tuple of prediction value and the according bounding box.
    If a ``threshold`` is specified, only those ``prediction``\s are returned, which exceed the given ``threshold``.

    .. note::
       The ``threshold`` does not overwrite the cascade thresholds `:py:attr:`Cascade.thresholds`, but only threshold the final prediction.
       Specifying the ``threshold`` here is just slightly faster than thresholding the yielded prediction.

    **Parameters:**

    ``cascade`` : :py:class:`Cascade`
      The cascade that performs the predictions

    ``image`` : array_like(2D)
      The image for which the predictions should be computed

    ``threshold`` : float
      The threshold, which limits the number of predictions

    **Yields:**

    ``prediction`` : float
      The prediction value for the current bounding box

    ``bounding_box`` : :py:class:`BoundingBox`
      An iterator over all possible sampled bounding boxes (which exceed the prediction ``threshold``, if given)
    """

    for scale, scaled_image_shape in self.scales(image):
      # prepare the feature extractor to extract features from the given image
      cascade.prepare(image, scale)
      for bb in self.sample_scaled(scaled_image_shape):
        # return the prediction and the bounding box, if the prediction is over threshold
        prediction = cascade(bb)
        if threshold is None or prediction > threshold:
          yield prediction, bb.scale(1./scale)
