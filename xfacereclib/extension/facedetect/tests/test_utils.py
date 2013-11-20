import unittest
import math
from nose.plugins.skip import SkipTest

import bob
import facereclib
import pkg_resources

import xfacereclib.extension.facedetect

class UtilsTests(unittest.TestCase):

  def test01_bounding_box_py(self):
    # Tests that the bounding box calculation works as expected

    # check the indirect ways using eye coordinates
    bb = xfacereclib.extension.facedetect.utils.BoundingBox(leye=(10,10), reye=(10,30), padding={'left':-1, 'right':1, 'top':-1, 'bottom':1})
    self.assertEqual(bb.top, -10)
    self.assertEqual(bb.bottom, 29)
    self.assertEqual(bb.left, 0)
    self.assertEqual(bb.right, 39)

    # check the scaling functionality
    sbb = bb.scale(0.5)
    self.assertEqual(sbb.top, -5)
    self.assertEqual(sbb.bottom, 14)
    self.assertEqual(sbb.left, 0)
    self.assertEqual(sbb.right, 19)
    sbb = bb.scale(0.84)
    self.assertEqual(sbb.top, -8)
    self.assertEqual(sbb.bottom, 25)
    self.assertEqual(sbb.left, 0)
    self.assertEqual(sbb.right, 33)


    bb = xfacereclib.extension.facedetect.utils.BoundingBox(leye=(10,10), reye=(10,30))
    self.assertEqual(bb.top, -4)
    self.assertEqual(bb.bottom, 43)
    self.assertEqual(bb.left, 0)
    self.assertEqual(bb.right, 39)

    # test that left and right profile versions work
    lbb = xfacereclib.extension.facedetect.utils.BoundingBox(source='left-profile', mouth=(40,10), eye=(20,10))
    self.assertEqual(bb.top, -4)
    self.assertEqual(bb.bottom, 43)
    self.assertEqual(bb.left, 0)
    self.assertEqual(bb.right, 39)

    # test the direct way
    bb1 = xfacereclib.extension.facedetect.utils.BoundingBox(topleft=(10,20), bottomright=(29,39))
    self.assertEqual(bb1.top, 10)
    self.assertEqual(bb1.bottom, 29)
    self.assertEqual(bb1.left, 20)
    self.assertEqual(bb1.right, 39)
    self.assertEqual(bb1.area(), 400)

    bb2 = xfacereclib.extension.facedetect.utils.BoundingBox(topleft=(15,25), bottomright=(34,44))
    bb3 = bb1.overlap(bb2)
    self.assertEqual(bb3.top, 15)
    self.assertEqual(bb3.bottom, 29)
    self.assertEqual(bb3.left, 25)
    self.assertEqual(bb3.right, 39)
    self.assertEqual(bb3.area(), 225)

    # check the similarity function
    self.assertEqual(bb1.similarity(bb1), 1.)
    self.assertEqual(bb3.similarity(bb2), 0.5625)
    self.assertEqual(bb3.similarity(bb1), bb1.similarity(bb3))


  def test02_bounding_box_cpp(self):
    # Tests that the bounding box calculation works as expected

    # check the indirect ways using eye coordinates
    bb = xfacereclib.extension.facedetect.utils.bounding_box_from_annotation(leye=(10,10), reye=(10,30), padding={'left':-1, 'right':1, 'top':-1, 'bottom':1})
    self.assertEqual(bb.top, -10)
    self.assertEqual(bb.bottom, 29)
    self.assertEqual(bb.height, 40)
    self.assertEqual(bb.left, 0)
    self.assertEqual(bb.right, 39)
    self.assertEqual(bb.width, 40)

    # check the scaling functionality
    sbb = bb.scale(0.5)
    self.assertEqual(sbb.top, -5)
    self.assertEqual(sbb.bottom, 14)
    self.assertEqual(sbb.left, 0)
    self.assertEqual(sbb.right, 19)
    sbb = bb.scale(0.84)
    self.assertEqual(sbb.top, -8)
    self.assertEqual(sbb.bottom, 25)
    self.assertEqual(sbb.left, 0)
    self.assertEqual(sbb.right, 33)


    bb = xfacereclib.extension.facedetect.utils.bounding_box_from_annotation(leye=(10,10), reye=(10,30))
    self.assertEqual(bb.top, -4)
    self.assertEqual(bb.bottom, 43)
    self.assertEqual(bb.left, 0)
    self.assertEqual(bb.right, 39)

    # test that left and right profile versions work
    lbb = xfacereclib.extension.facedetect.utils.bounding_box_from_annotation(source='left-profile', mouth=(40,10), eye=(20,10))
    self.assertEqual(bb.top, -4)
    self.assertEqual(bb.bottom, 43)
    self.assertEqual(bb.left, 0)
    self.assertEqual(bb.right, 39)

    # test the direct way
    bb1 = xfacereclib.extension.facedetect.utils.bounding_box_from_annotation(topleft=(10,20), bottomright=(29,39))
    self.assertEqual(bb1.top, 10)
    self.assertEqual(bb1.bottom, 29)
    self.assertEqual(bb1.left, 20)
    self.assertEqual(bb1.right, 39)
    self.assertEqual(bb1.area, 400)

    bb2 = xfacereclib.extension.facedetect.utils.bounding_box_from_annotation(topleft=(15,25), bottomright=(34,44))
    bb3 = bb1.overlap(bb2)
    self.assertEqual(bb3.top, 15)
    self.assertEqual(bb3.bottom, 29)
    self.assertEqual(bb3.left, 25)
    self.assertEqual(bb3.right, 39)
    self.assertEqual(bb3.area, 225)

    # check the similarity function
    self.assertEqual(bb1.similarity(bb1), 1.)
    self.assertEqual(bb3.similarity(bb2), 0.5625)
    self.assertEqual(bb3.similarity(bb1), bb1.similarity(bb3))

