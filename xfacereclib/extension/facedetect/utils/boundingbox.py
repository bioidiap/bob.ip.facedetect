import bob
import facereclib
import math

from .._features import BoundingBox as CppBoundingBox


available_sources = {
  'direct'        : ('topleft', 'bottomright'),
  'eyes'          : ('leye', 'reye'),
  'left-profile'  : ('eye', 'mouth'),
  'right-profile' : ('eye', 'mouth'),
  'ellipse'       : ('center', 'angle', 'axis_radius')
}

# This struct specifies, which paddings should be applied to which source.
# All values are relative to the inter-node distance
default_paddings = {
  'direct'        : None,
  'eyes'          : {'left' : -1.0, 'right' : +1.0, 'top': -0.7, 'bottom' : 1.7}, # These parameters are used to match Cosmin's implementation (which was buggy...)
  'left-profile'  : {'left' : -0.2, 'right' : +0.8, 'top': -1.0, 'bottom' : 1.0},
  'right-profile' : {'left' : -0.8, 'right' : +0.2, 'top': -1.0, 'bottom' : 1.0},
  'ellipse'       : None
}


def bounding_box_from_annotation(source=None, padding=None, **kwargs):
  """Creates a bounding box from the given parameters.
  If 'sources' are specified, the according keywords (see available_sources) must be given as well.
  Otherwise, the source is estimated from the given keyword parameters if possible.

  If 'topleft' and 'bottomright' are given (i.e., the 'direct' source), they are taken as is.
  Note that the 'bottomright' is NOT included in the bounding box.

  For source 'ellipse', the bounding box is computed to capture the whole ellipse, even if it is rotated.

  For other sources (i.e., 'eyes'), the center of the two given positions is computed, and the 'padding' is applied.
  If 'padding ' is None (the default) then the default_paddings of this source are used instead.
  """

  if source is None:
    # try to estimate the source
    for s,k in available_sources.iteritems():
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
    facereclib.utils.debug("Estimated source '%s' since the keywords '%s' and '%s' are given" % (source, available_sources[source][0], available_sources[source][1]))

  assert source in available_sources

  # use default padding if not specified
  if padding is None:
    padding = default_paddings[source]

  keys = available_sources[source]
  if source == 'ellipse':
    # compute the tight bounding box for the ellipse
    angle = kwargs['angle']
    axis = kwargs['axis_radius']
    center = kwargs['center']
    dx = abs(math.cos(angle) * axis[0]) + abs(math.sin(angle) * axis[1])
    dy = abs(math.sin(angle) * axis[0]) + abs(math.cos(angle) * axis[1])
    top = center[0] - dy
    bottom = center[0] + dy - 1
    left = center[1] - dx
    right = center[1] + dx - 1
  elif padding is None:
    # There is no padding to be applied -> take nodes as they are
    top    = kwargs[keys[0]][0]
    bottom = kwargs[keys[1]][0]
    left   = kwargs[keys[0]][1]
    right  = kwargs[keys[1]][1]
  else:
    from . import sqr, irnd
    # apply padding
    pos_0 = kwargs[keys[0]]
    pos_1 = kwargs[keys[1]]
    tb_center = float(pos_0[0] + pos_1[0]) / 2.
    lr_center = float(pos_0[1] + pos_1[1]) / 2.
    distance = math.sqrt(sqr(pos_0[0] - pos_1[0]) + sqr(pos_0[1] - pos_1[1]))

    top    = irnd(tb_center + padding['top'] * distance)
    bottom = irnd(tb_center + padding['bottom'] * distance - 1)
    left   = irnd(lr_center + padding['left'] * distance)
    right  = irnd(lr_center + padding['right'] * distance - 1)

  return CppBoundingBox(top, left, bottom - top + 1, right - left + 1)


class BoundingBox:

  def __init__(self, source=None, padding=None, **kwargs):
    """Creates a bounding box from the given parameters.
    If 'sources' are specified, the according keywords (see available_sources) must be given as well.
    Otherwise, the source is estimated from the given keyword parameters if possible.

    If 'topleft' and 'bottomright' are given (i.e., the 'direct' source), they are taken as is.
    Note that the 'bottomright' is NOT included in the bounding box.

    For source 'ellipse', the bounding box is computed to capture the whole ellipse, even if it is rotated.

    For other sources (i.e., 'eyes'), the center of the two given positions is computed, and the 'padding' is applied.
    If 'padding ' is None (the default) then the default_paddings of this source are used instead.
    """

    if source is None:
      # try to estimate the source
      for s,k in available_sources.iteritems():
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
      facereclib.utils.debug("Estimated source '%s' since the keywords '%s' and '%s' are given" % (source, available_sources[source][0], available_sources[source][1]))

    assert source in available_sources

    # use default padding if not specified
    if padding is None:
      padding = default_paddings[source]

    keys = available_sources[source]
    if source == 'ellipse':
      # compute the tight bounding box for the ellipse
      angle = kwargs['angle']
      axis = kwargs['axis_radius']
      center = kwargs['center']
      dx = abs(math.cos(angle) * axis[0]) + abs(math.sin(angle) * axis[1])
      dy = abs(math.sin(angle) * axis[0]) + abs(math.cos(angle) * axis[1])
      self.top = center[0] - dy
      self.bottom = center[0] + dy
      self.left = center[1] - dx
      self.right = center[1] + dx
    elif padding is None:
      # There is no padding to be applied -> take nodes as they are
      self.top    = kwargs[keys[0]][0]
      self.bottom = kwargs[keys[1]][0]
      self.left   = kwargs[keys[0]][1]
      self.right  = kwargs[keys[1]][1]
    else:
      from . import sqr, irnd
      # apply padding
      pos_0 = kwargs[keys[0]]
      pos_1 = kwargs[keys[1]]
      tb_center = float(pos_0[0] + pos_1[0]) / 2.
      lr_center = float(pos_0[1] + pos_1[1]) / 2.
      distance = math.sqrt(sqr(pos_0[0] - pos_1[0]) + sqr(pos_0[1] - pos_1[1]))

      self.top    = irnd(tb_center + padding['top'] * distance)
      self.bottom = irnd(tb_center + padding['bottom'] * distance - 1)
      self.left   = irnd(lr_center + padding['left'] * distance)
      self.right  = irnd(lr_center + padding['right'] * distance - 1)


  def __str__(self):
    """Returns a string representation of this bounding box."""
    return "<BoundingBox: top=%d, left=%d, bottom=%d, right=%d>" % (self.top, self.left, self.bottom, self.right)


  def scale(self, scale):
    """Returns a scaled version of this bounding box."""
    from . import irnd
    # To match the rounding in the C++ implementation
    height = self.bottom - self.top + 1.
    width = self.right - self.left + 1.
    return BoundingBox("direct", topleft=(irnd(self.top*scale), irnd(self.left*scale)), bottomright=(irnd(self.top*scale) + irnd(height*scale)-1, irnd(self.left*scale) + irnd(width*scale)-1))


  def shift(self, dy, dx):
    """Returns a shifted version of this bounding box."""
    return BoundingBox("direct", topleft=(self.top+dy, self.left+dx), bottomright=(self.bottom+dy, self.right+dx))

  def extract(self, image):
    """Returns a sub-window of the given image that corresponds to the bounding box."""
    return image[self.top : self.bottom, self.left : self.right]


  def overlap(self, other):
    """Computes the intersection between this bounding box and the given one.
    If the two bounding boxes to not overlap, None is returned."""
    top    = max(self.top, other.top)
    bottom = min(self.bottom, other.bottom)
    left   = max(self.left, other.left)
    right  = min(self.right, other.right)

    if top > bottom or left > right:
      return None

    return BoundingBox('direct', topleft=(top, left), bottomright=(bottom, right))


  def area(self):
    """Computes the area that is spanned by this bounding box."""
    return (self.right - self.left + 1) * (self.bottom - self.top + 1)


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

  def draw(self, image, color):
    """Draws the bounding box into the given image using the given color."""
    bob.ip.draw_box(image, self.left, self.top, self.right - self.left + 1, self.bottom - self.top + 1, color)


def prune(detections, threshold=None):
  """Removes overlapping detections, using the given detection overlap threshold."""

  # first, sort the detections based on their detection value
  sorted_detections = sorted(detections, cmp=lambda x,y: cmp(x[0], y[0]), reverse=True)

  if threshold is None:
    return sorted_detections

  # now, add detections as long as they don't overlap with previous detections
  pruned_detections = []
  for value, detection in sorted_detections:
    add = True
    # check if an overlap with previously added detections is found
    for _, pruned in pruned_detections:
      if detection.similarity(pruned) > threshold:
        # found overlap, so don't add
        add = False
        break;
    if add:
      pruned_detections.append((value, detection))

  return pruned_detections

