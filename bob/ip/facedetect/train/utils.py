import numpy
import math
import os

from .._library import BoundingBox


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
  """bounding_box_from_annotation(source, padding, **kwargs) -> bounding_box

  Creates a bounding box from the given parameters, which are, in general, annotations read using :py:func:`bob.ip.facedetect.read_annotation_file`.
  Different kinds of annotations are supported, given by the ``source`` keyword:

  * ``direct`` : bounding boxes are directly specified by keyword arguments ``topleft`` and ``bottomright``
  * ``eyes`` : the left and right eyes are specified by keyword arguments ``leye`` and ``reye``
  * ``left-profile`` : the left eye and the mouth are specified by keyword arguments ``eye`` and ``mouth``
  * ``right-profile`` : the right eye and the mouth are specified by keyword arguments ``eye`` and ``mouth``
  * ``ellipse`` : the face ellipse as well as face angle and axis radius is provided by keyword arguments ``center``, ``angle`` and ``axis_radius``

  If a ``source`` is specified, the according keywords must be given as well.
  Otherwise, the source is estimated from the given keyword parameters if possible.

  If 'topleft' and 'bottomright' are given (i.e., the 'direct' source), they are taken as is.
  Note that the 'bottomright' is NOT included in the bounding box.
  Please assure that the aspect ratio of the bounding box is 6:5 (height : width).

  For source 'ellipse', the bounding box is computed to capture the whole ellipse, even if it is rotated.

  For other sources (i.e., 'eyes'), the center of the two given positions is computed, and the ``padding`` is applied, which is relative to the distance between the two given points.
  If ``padding`` is ``None`` (the default) the default_paddings of this source are used instead.
  These padding is required to keep an aspect ratio of 6:5.

  **Parameters:**

  ``source`` : str or ``None``
    The type of annotations present in the list of keyword arguments, see above.

  ``padding`` : {'top':float, 'bottom':float, 'left':float, 'right':float}
    This padding is added to the center between the given points, to define the top left and bottom right positions in the bounding box; values are relative to the distance between the two given points; ignored for some of the ``source``\s

  ``kwargs`` : key=value
    Further keyword arguments specifying the annotations.

  **Returns:**

  bounding_box : :py:class:`BoundingBox`
    The bounding box that was estimated from the given annotations.
  """

  if source is None:
    # try to estimate the source
    for s,k in available_sources.items():
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
    bottom = center[0] + dy
    left = center[1] - dx
    right = center[1] + dx
  elif padding is None:
    # There is no padding to be applied -> take nodes as they are
    top    = kwargs[keys[0]][0]
    bottom = kwargs[keys[1]][0]
    left   = kwargs[keys[0]][1]
    right  = kwargs[keys[1]][1]
  else:
    # apply padding
    pos_0 = kwargs[keys[0]]
    pos_1 = kwargs[keys[1]]
    tb_center = float(pos_0[0] + pos_1[0]) / 2.
    lr_center = float(pos_0[1] + pos_1[1]) / 2.
    distance = math.sqrt((pos_0[0] - pos_1[0])**2 + (pos_0[1] - pos_1[1])**2)

    top    = tb_center + padding['top'] * distance
    bottom = tb_center + padding['bottom'] * distance
    left   = lr_center + padding['left'] * distance
    right  = lr_center + padding['right'] * distance

  return BoundingBox((top, left), (bottom - top, right - left))


def expected_eye_positions(bounding_box, padding = None):
  """expected_eye_positions(bounding_box, padding) -> eyes

  Computes the expected eye positions based on the relative coordinates of the bounding box.

  This function can be used to translate between bounding-box-based image cropping and eye-location-based alignment.
  The returned eye locations return the **average** eye locations, no landmark detection is performed.

  **Parameters:**

  ``bounding_box`` : :py:class:`BoundingBox`
    The face bounding box as detected by one of the functions in ``bob.ip.facedetect``.

  ``padding`` : {'top':float, 'bottom':float, 'left':float, 'right':float}
    The padding that was used for the ``eyes`` source in :py:func:`bounding_box_from_annotation`, has a proper default.

  **Returns:**

  ``eyes`` : {'reye' : (rey, rex), 'leye' : (ley, lex)}
    A dictionary containing the average left and right eye annotation.
  """
  if padding is None:
    padding = default_paddings['eyes']
  top, left, right = padding['top'], padding['left'], padding['right']
  inter_eye_distance = (bounding_box.size[1]) / (right - left)
  return {
    'reye':(bounding_box.top_f - top*inter_eye_distance, bounding_box.left_f - left/2.*inter_eye_distance),
    'leye':(bounding_box.top_f - top*inter_eye_distance, bounding_box.right_f - right/2.*inter_eye_distance)
  }



def parallel_part(data, parallel):
  """parallel_part(data, parallel) -> part

  Splits off samples from the the given data list and the given number of parallel jobs based on the ``SGE_TASK_ID`` environment variable.

  **Parameters:**

  ``data`` : [object]
    A list of data that should be split up into ``parallel`` parts

  ``parallel`` : int or ``None``
    The total number of parts, in which the data should be split into

  **Returns:**

  ``part`` : [object]
    The desired partition of the ``data``
  """
  if parallel is None or "SGE_TASK_ID" not in os.environ:
    return data

  data_per_job = int(math.ceil(float(len(data)) / float(parallel)))
  task_id = int(os.environ['SGE_TASK_ID'])
  first = (task_id-1) * data_per_job
  last = min(len(data), task_id * data_per_job)
  return data[first:last]



def quasi_random_indices(number_of_total_items, number_of_desired_items = None):
  """quasi_random_indices(number_of_total_items, [number_of_desired_items]) -> index

  Yields an iterator to a quasi-random list of indices that will contain exactly the number of desired indices (or the number of total items in the list, if this is smaller).

  This function can be used to retrieve a consistent and reproducible list of indices of the data, in case the ``number_of_total_items`` is lower that the given ``number_of_desired_items``.

  **Parameters:**

  ``number_of_total_items`` : int
    The total number of elements in the collection, which should be sub-sampled

  ``number_of_desired_items`` : int or ``None``
    The number of items that should be used; if ``None`` or greater than ``number_of_total_items``, all indices are yielded

  **Yields:**

  ``index`` : int
    An iterator to indices, which will span ``number_of_total_items`` evenly.
  """
  # check if we need to compute a sublist at all
  if number_of_desired_items is None or number_of_desired_items >= number_of_total_items or number_of_desired_items < 0:
    for i in range(number_of_total_items):
      yield i
  else:
    increase = float(number_of_total_items)/float(number_of_desired_items)
    # generate a regular quasi-random index list
    for i in range(number_of_desired_items):
      yield int((i +.5)*increase)
