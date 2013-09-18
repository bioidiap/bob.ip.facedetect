import unittest
import math
from nose.plugins.skip import SkipTest

import bob
import facereclib
import pkg_resources

import xfacereclib.extension.facedetect

class UtilsTests(unittest.TestCase):

  def test01_bounding_box(self):
    # Tests that the bounding box calculation works as expected

    # check the indirect ways using eye coordinates
    bb = xfacereclib.extension.facedetect.utils.BoundingBox(leye=(10,10), reye=(10,30), padding={'left':-1, 'right':1, 'top':-1, 'bottom':1})
    self.assertEqual(bb.m_top, -10)
    self.assertEqual(bb.m_bottom, 31)
    self.assertEqual(bb.m_left, 0)
    self.assertEqual(bb.m_right, 41)

    # check the scaling functionality
    sbb = bb.scale(0.5)
    self.assertEqual(sbb.m_top, -5)
    self.assertEqual(sbb.m_bottom, 16)
    self.assertEqual(sbb.m_left, 0)
    self.assertEqual(sbb.m_right, 21)


    bb = xfacereclib.extension.facedetect.utils.BoundingBox(leye=(10,10), reye=(10,30))
    self.assertEqual(bb.m_top, -4)
    self.assertEqual(bb.m_bottom, 45)
    self.assertEqual(bb.m_left, 0)
    self.assertEqual(bb.m_right, 41)

    # test that left and right profile versions work
    lbb = xfacereclib.extension.facedetect.utils.BoundingBox(source='left-profile', mouth=(40,10), eye=(20,10))
    self.assertEqual(bb.m_top, -4)
    self.assertEqual(bb.m_bottom, 45)
    self.assertEqual(bb.m_left, 0)
    self.assertEqual(bb.m_right, 41)

    # test the direct way
    bb1 = xfacereclib.extension.facedetect.utils.BoundingBox(topleft=(10,20), bottomright=(30,40))
    self.assertEqual(bb1.m_top, 10)
    self.assertEqual(bb1.m_bottom, 30)
    self.assertEqual(bb1.m_left, 20)
    self.assertEqual(bb1.m_right, 40)
    self.assertEqual(bb1.area(), 400)

    bb2 = xfacereclib.extension.facedetect.utils.BoundingBox(topleft=(15,25), bottomright=(35,45))
    bb3 = bb1.overlap(bb2)
    self.assertEqual(bb3.m_top, 15)
    self.assertEqual(bb3.m_bottom, 30)
    self.assertEqual(bb3.m_left, 25)
    self.assertEqual(bb3.m_right, 40)
    self.assertEqual(bb3.area(), 225)

    # check the similarity function
    self.assertEqual(bb1.similarity(bb1), 1.)
    self.assertEqual(bb3.similarity(bb2), 0.5625)
    self.assertEqual(bb3.similarity(bb1), bb1.similarity(bb3))

