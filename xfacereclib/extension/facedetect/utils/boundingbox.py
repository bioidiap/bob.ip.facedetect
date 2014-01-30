import bob
import numpy
import facereclib
import math

from .._features import BoundingBox, overlapping_detections


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
    from . import sqr
    # apply padding
    pos_0 = kwargs[keys[0]]
    pos_1 = kwargs[keys[1]]
    tb_center = float(pos_0[0] + pos_1[0]) / 2.
    lr_center = float(pos_0[1] + pos_1[1]) / 2.
    distance = math.sqrt(sqr(pos_0[0] - pos_1[0]) + sqr(pos_0[1] - pos_1[1]))

    top    = tb_center + padding['top'] * distance
    bottom = tb_center + padding['bottom'] * distance - 1
    left   = lr_center + padding['left'] * distance
    right  = lr_center + padding['right'] * distance - 1

  return BoundingBox(top, left, bottom - top + 1, right - left + 1)


def expected_eye_positions(bounding_box):
  """Computes the expected eye positions based on the relative coordinates of the bounding box."""
  top, left, right = default_paddings['eyes']['top'], default_paddings['eyes']['left'], default_paddings['eyes']['right']
  inter_eye_distance = (bounding_box.width) / (right - left)
  return {
    'reye':(bounding_box.top - top*inter_eye_distance, bounding_box.left - left/2.*inter_eye_distance),
    'leye':(bounding_box.top - top*inter_eye_distance, bounding_box.right - right/2.*inter_eye_distance)
  }



def best_detection(detections, predictions, minimum_overlap = 0.2):
  # remove all negative predictions since they harm the calculation of the weights
  detections = [detections[i] for i in range(len(detections)) if predictions[i] > 0]
  predictions = [predictions[i] for i in range(len(detections)) if predictions[i] > 0]

  # keep only the bounding boxes with the highest overlap
  detections, predictions = overlapping_detections(detections, numpy.array(predictions), minimum_overlap)

  # compute the mean of the detected bounding boxes
  s = sum(predictions)
  weights = [p/s for p in predictions]
  top = sum(w * b.top_f for w, b in zip(weights, detections))
  left = sum(w * b.left_f for w, b in zip(weights, detections))
  bottom = sum(w * b.bottom_f for w, b in zip(weights, detections))
  right = sum(w * b.right_f for w, b in zip(weights, detections))

#  value = sum(w*p for w,p in zip(weights, predictions))
  # as the detection value, we use the *BEST* value of all detections.
  value = predictions[0]

  return BoundingBox(top, left, bottom-top+1, right-left+1), value

