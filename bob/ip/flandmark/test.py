#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
# Andre Anjos <andre.anjos@idiap.ch>
# Wed 16 Apr 09:35:37 2014 CEST

"""Tests for flandmark python bindings
"""

import os
import numpy
import functools
import pkg_resources
import nose.tools
import bob.io.base
import bob.io.image
import bob.ip.color

from . import Flandmark

def F(f):
  """Returns the test file on the "data" subdirectory"""
  return pkg_resources.resource_filename(__name__, os.path.join('data', f))

LENA = F('lena.jpg')
LENA_BBX = [
    [214, 202, 183, 183]
    ]

MULTI = F('multi.jpg')
MULTI_BBX = [
    [326, 20, 31, 31],
    [163, 25, 34, 34],
    [253, 42, 28, 28],
    ]


def pnpoly(point, vertices):
  """Python translation of the C algorithm taken from:
  http://www.ecse.rpi.edu/Homepages/wrf/Research/Short_Notes/pnpoly.html
  """

  (x, y) = point
  j = vertices[-1]
  c = False
  for i in vertices:
    if ( (i[1] > y) != (j[1] > y) ) and \
        ( x < (((j[0]-i[0]) * (y-i[1]) / (j[1]-i[1])) + i[0]) ):
      c = not c
    j = i

  return c

def is_inside(point, box, eps=1e-5):
  """Calculates if a point lies inside a bounding box"""

  (y, x, height, width) = box
  #note: vertices must be organized clockwise
  vertices = numpy.array([
    (x-eps, y-eps),
    (x+width+eps, y-eps),
    (x+width+eps, y+height+eps),
    (x-eps, y+height+eps),
    ], dtype=float)
  return pnpoly((point[1], point[0]), vertices)


def test_is_inside():

  box = (0, 0, 1, 1)

  # really inside
  assert is_inside((0.5, 0.5), box, eps=1e-10)

  # on the limit of the box
  assert is_inside((0.0, 0.0), box, eps=1e-10)
  assert is_inside((1.0, 1.0), box, eps=1e-10)
  assert is_inside((1.0, 0.0), box, eps=1e-10)
  assert is_inside((0.0, 1.0), box, eps=1e-10)

def test_is_outside():

  box = (0, 0, 1, 1)

  # really outside the box
  assert not is_inside((1.5, 1.0), box, eps=1e-10)
  assert not is_inside((0.5, 1.5), box, eps=1e-10)
  assert not is_inside((1.5, 1.5), box, eps=1e-10)
  assert not is_inside((-0.5, -0.5), box, eps=1e-10)


def test_lena():

  img = bob.io.base.load(LENA)
  gray = bob.ip.color.rgb_to_gray(img)
  (x, y, width, height) = LENA_BBX[0]

  flm = Flandmark()
  keypoints = flm.locate(gray, y, x, height, width)
  nose.tools.eq_(keypoints.shape, (8, 2))
  nose.tools.eq_(keypoints.dtype, 'float64')
  for k in keypoints:
    assert is_inside(k, (y, x, height, width), eps=1)

def test_full_image():

  img = bob.io.base.load(LENA)
  gray = bob.ip.color.rgb_to_gray(img)

  flm = Flandmark()
  lm1 = flm.locate(gray, 0, 0, gray.shape[0], gray.shape[1])
  lm2 = flm.locate(gray)
  assert all(numpy.allclose(lm1[i],lm2[i]) for i in range(len(lm1)))



def test_multi():

  img = bob.io.base.load(MULTI)
  gray = bob.ip.color.rgb_to_gray(img)

  flm = Flandmark()
  for (x, y, width, height) in MULTI_BBX:
    keypoints = flm.locate(gray, y, x, height, width)
    nose.tools.eq_(keypoints.shape, (8, 2))
    nose.tools.eq_(keypoints.dtype, 'float64')
    for k in keypoints:
      assert is_inside(k, (y, x, height, width), eps=1)
