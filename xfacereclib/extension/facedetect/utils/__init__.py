from boundingbox import bounding_box_from_annotation, expected_eye_positions, best_detection
from database import training_image_annot, test_image_annot

def sqr(x):
  """This function computes the square of the given value.
  (I don't know why such a function is not part of the math module)."""
  return x*x

def irnd(x):
  """This function returns the integer value of the rounded given float value."""
  return int(round(x))


def detect_landmarks(localizer, image, bounding_box):
  scales = [1., 0.9, 0.8, 1.1, 1.2]
  shifts = [0, 0.1, 0.2, -0.1, -0.2]

  uint8_image = image.astype(numpy.uint8)

  for scale in scales:
    bs = bounding_box.scale_centered(scale)
    for y in shifts:
      by = bs.shift(y * bs.height, 0)
      for x in shifts:
        bb = by.shift(0, x * bs.width)

        top = max(bb.top, 0)
        left = int(max(bb.left, 0))
        bottom = min(bb.bottom, image.shape[0]-1)
        right = int(min(bb.right, image.shape[1]-1))
        landmarks = localizer.localize(uint8_image, top, left, bottom-top+1, right-left+1)

        if len(landmarks):
          facereclib.utils.debug("Found landmarks with scale %1.1f, and shift %1.1fx%1.1f" % (scale, y, x))
          return landmarks

  return []


