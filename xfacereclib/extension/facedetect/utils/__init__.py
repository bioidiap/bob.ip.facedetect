from boundingbox import BoundingBox

from ..detector import LBPFeatures, MBLBPFeatures
import bob


def sqr(x):
  """This function computes the square of the given value.
  (I don't know why such a function is not part of the math module)."""
  return x*x

def irnd(x):
  """This function returns the integer value of the rounded given float value."""
  return int(round(x))

