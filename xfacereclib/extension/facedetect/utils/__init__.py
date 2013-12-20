from boundingbox import bounding_box_from_annotation, expected_eye_positions, best_detection
from database import training_image_annot, test_image_annot

def sqr(x):
  """This function computes the square of the given value.
  (I don't know why such a function is not part of the math module)."""
  return x*x

def irnd(x):
  """This function returns the integer value of the rounded given float value."""
  return int(round(x))


