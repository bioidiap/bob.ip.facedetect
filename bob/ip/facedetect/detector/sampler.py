
import math
from .._library import BoundingBox

import bob.ip.base


class Sampler:
  """This class generates (samples) bounding boxes for different scales and locations in the image.

  It computes different scales of the image and provides a tight set of :py:class:`BoundingBox` of a given patch size for the given image.

  **Constructor Documentation:**

    Generates a patch-sampler, which will scan images and sample bounding boxes.

    Parameters:

    patch_size : (int, int)
      the size of the patch (i.e., the bounding box) to sample

    scale_factor : float
      image pyramids are computed using the given scale factor between two scales

    lowest_scale : float or None
      patches which will be lower than the given scale times the image resolution will not be taken into account;
      if 0. all possible patches will be considered

    distance : int
      the distance in both horizontal and vertical direction to generate samples
  """

  def __init__(self, patch_size = (24,20), scale_factor = math.pow(2., -1./16.), lowest_scale = math.pow(2., -6.), distance = 2):

    self.m_patch_box = BoundingBox((0, 0), patch_size)
    self.m_scale_factor = scale_factor
    self.m_lowest_scale = lowest_scale
    self.m_distance = distance


  def scales(self, image):
    """Computes the all possible scales for the given image and returns a tuple of the scale and the scaled image shape as an iterator."""
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


  def sample_scaled(self, scaled_image_shape):
    """Returns an iterator that iterates over all sampled bounding boxes in the given (scaled) image shape."""
    for y in range(0, scaled_image_shape[-2]-self.m_patch_box.bottomright[0], self.m_distance):
      for x in range(0, scaled_image_shape[-1]-self.m_patch_box.bottomright[1], self.m_distance):
        # create bounding box for the current shift
        yield self.m_patch_box.shift((y,x))

  def sample(self, image):
    """Returns an iterator over all bounding boxes that are sampled for the given image."""
    for scale, scaled_image_shape in self.scales(image):
      # prepare the feature extractor to extract features from the given image
      for bb in self.sample_scaled(scaled_image_shape):
        # extract features for
        yield bb.scale(1./scale)


  def iterate(self, image, feature_extractor, feature_vector):
    """Scales the given image and extracts bounding boxes, computes the features for the given feature extractor and returns an ITERATOR returning a the bounding_box.
    """
    for scale, scaled_image_shape in self.scales(image):
      # prepare the feature extractor to extract features from the given image
      feature_extractor.prepare(image, scale)
      for bb in self.sample_scaled(scaled_image_shape):
        # extract features for
        feature_extractor.extract_indexed(bb, feature_vector)
        yield bb.scale(1./scale)


  def iterate_cascade(self, cascade, image, threshold = None):
    """Iterates over the given image and computes the cascade of classifiers."""

    for scale, scaled_image_shape in self.scales(image):
      # prepare the feature extractor to extract features from the given image
      cascade.prepare(image, scale)
      for bb in self.sample_scaled(scaled_image_shape):
        # return the prediction and the bounding box, if the prediction is over threshold
        prediction = cascade(bb)
        if threshold is None or prediction > threshold:
          yield prediction, bb.scale(1./scale)
