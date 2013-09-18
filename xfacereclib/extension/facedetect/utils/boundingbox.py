

import facereclib
import math

class BoundingBox:

  available_sources = {
    'direct'        : ('topleft', 'bottomright'),
    'eyes'          : ('leye', 'reye'),
    'left-profile'  : ('eye', 'mouth'),
    'right-profile' : ('eye', 'mouth')
  }

  # This struct specifies, which paddings should be applied to which source.
  # All values are relative to the inter-node distance
  default_paddings = {
    'direct'        : None,
    'eyes'          : {'left' : -1.0, 'right' : +1.0, 'top': -0.7, 'bottom' : 1.7}, # These parameters are used to match Cosmin's implementation (which was buggy...)
    'left-profile'  : {'left' : -0.2, 'right' : +0.8, 'top': -1.0, 'bottom' : 1.0},
    'right-profile' : {'left' : -0.8, 'right' : +0.2, 'top': -1.0, 'bottom' : 1.0}
  }

  def __init__(self, source=None, padding=None, **kwargs):
    """Creates a bounding box from the given parameters.
    If 'sources' are specified, the according keywords (see available_sources) must be given as well.
    Otherwise, the source is estimated from the given keyword parameters if possible.

    If 'topleft' and 'bottomright' are given (i.e., the 'direct' source), they are taken as is.
    Note that the 'bottomright' is NOT included in the bounding box.

    For other sources (i.e., 'eyes'), the center of the two given positions is computed, and the 'padding' is applied.
    If 'padding ' is None (the default) then the default_paddings of this source are used instead.
    """

    if source is None:
      # try to estimate the source
      for s,k in self.available_sources.iteritems():
        # check if the according keyword arguments are given
        if k[0] in kwargs and k[1] in kwargs:
          # check if we already assigned a source before
          if source is not None:
            raise ValueError("The given list of keywords (%s) is ambiguous. Please specify a source" % kwargs)
          # assign source
          source = s

      # check if a source could be estimated from the keywords
      if source is None:
        raise ValueError("The given list of keywords (%s) could not be interpreted" % kwargs)
      facereclib.utils.debug("Estimated source '%s' since the keywords '%s' and '%s' are given" % (source, self.available_sources[source][0], self.available_sources[source][1]))

    assert source in self.available_sources

    # use default padding if not specified
    if padding is None:
      padding = self.default_paddings[source]

    keys = self.available_sources[source]
    if padding is None:
      # There is no padding to be applied -> take nodes as they are
      self.m_top    = kwargs[keys[0]][0]
      self.m_bottom = kwargs[keys[1]][0]
      self.m_left   = kwargs[keys[0]][1]
      self.m_right  = kwargs[keys[1]][1]
    else:
      from . import sqr, irnd
      # apply padding
      pos_0 = kwargs[keys[0]]
      pos_1 = kwargs[keys[1]]
      tb_center = float(pos_0[0] + pos_1[0]) / 2.
      lr_center = float(pos_0[1] + pos_1[1]) / 2.
      distance = math.sqrt(sqr(pos_0[0] - pos_1[0]) + sqr(pos_0[1] - pos_1[1]))

      self.m_top    = irnd(tb_center + padding['top'] * distance)
      self.m_bottom = irnd(tb_center + padding['bottom'] * distance + 1.)
      self.m_left   = irnd(lr_center + padding['left'] * distance)
      self.m_right  = irnd(lr_center + padding['right'] * distance + 1.)


  def __str__(self):
    """Returns a string representation of this bounding box."""
    return "<BoundingBox: top=%d, left=%d, bottom=%d, right=%d>" % (self.m_top, self.m_left, self.m_bottom, self.m_right)


  def scale(self, scale):
    """Returns a scaled version of this bounding box."""
    from . import irnd
    return BoundingBox("direct", topleft=(irnd(self.m_top*scale), irnd(self.m_left*scale)), bottomright=(irnd((self.m_bottom-1)*scale+1), irnd((self.m_right-1)*scale+1)))


  def shift(self, dy, dx):
    """Returns a shifted version of this bounding box."""
    return BoundingBox("direct", topleft=(self.m_top+dy, self.m_left+dx), bottomright=(self.m_bottom+dy, self.m_right+dx))

  def extract(self, image):
    """Returns a sub-window of the given image that corresponds to the bounding box."""
    return image[self.m_top : self.m_bottom, self.m_left : self.m_right]


  def overlap(self, other):
    """Computes the intersection between this bounding box and the given one.
    If the two bounding boxes to not overlap, None is returned."""
    top    = max(self.m_top, other.m_top)
    bottom = min(self.m_bottom, other.m_bottom)
    left   = max(self.m_left, other.m_left)
    right  = min(self.m_right, other.m_right)

    if top > bottom or left > right:
      return None

    return BoundingBox('direct', topleft=(top, left), bottomright=(bottom, right))


  def area(self):
    """Computes the area that is spanned by this bounding box."""
    return (self.m_right - self.m_left) * (self.m_bottom - self.m_top)


  def similarity(self, other):
    """Computes the similarity (i.e., the relative overlap) between this and the given bounding box.
    This formula is copied from Cosmins code, see the overlap function in Visioner/src/vision/vision.h and results in values between 0 and 1."""
    intersection = self.overlap(other)
    if intersection is None:
      # No intersection, so the bounding boxes are dissimilar
      return 0.
    else:
      # compute the similarity between the two rectangles as the relative area that the overlap captures
      return float(intersection.area()) / float(self.area() + other.area() - intersection.area())
