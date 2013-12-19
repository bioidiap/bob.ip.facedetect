import unittest
import math
from nose.plugins.skip import SkipTest

import bob
import numpy
import facereclib
import pkg_resources

import xfacereclib.extension.facedetect as fd

class UtilsTests(unittest.TestCase):

  def test01_bounding_box_py(self):
    # Tests that the bounding box calculation works as expected

    # check the indirect ways using eye coordinates
    bb = fd.utils.BoundingBox(leye=(10,10), reye=(10,30), padding={'left':-1, 'right':1, 'top':-1, 'bottom':1})
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


    bb = fd.utils.BoundingBox(leye=(10,10), reye=(10,30))
    self.assertEqual(bb.top, -4)
    self.assertEqual(bb.bottom, 43)
    self.assertEqual(bb.left, 0)
    self.assertEqual(bb.right, 39)

    # test that left and right profile versions work
    lbb = fd.utils.BoundingBox(source='left-profile', mouth=(40,10), eye=(20,10))
    self.assertEqual(bb.top, -4)
    self.assertEqual(bb.bottom, 43)
    self.assertEqual(bb.left, 0)
    self.assertEqual(bb.right, 39)

    # test the direct way
    bb1 = fd.utils.BoundingBox(topleft=(10,20), bottomright=(29,39))
    self.assertEqual(bb1.top, 10)
    self.assertEqual(bb1.bottom, 29)
    self.assertEqual(bb1.left, 20)
    self.assertEqual(bb1.right, 39)
    self.assertEqual(bb1.area(), 400)

    bb2 = fd.utils.BoundingBox(topleft=(15,25), bottomright=(34,44))
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
    bb = fd.utils.bounding_box_from_annotation(leye=(10,10), reye=(10,30), padding={'left':-1, 'right':1, 'top':-1, 'bottom':1})
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
    sbb = bb.scale_centered(2.)
    self.assertEqual(sbb.top, -30)
    self.assertEqual(sbb.bottom, 49)
    self.assertEqual(sbb.left, -20)
    self.assertEqual(sbb.right, 59)
    sbb = bb.scale(0.84)
    self.assertEqual(sbb.top, -8)
    self.assertEqual(sbb.bottom, 24)
    self.assertEqual(sbb.left, 0)
    self.assertEqual(sbb.right, 33)


    bb = fd.utils.bounding_box_from_annotation(leye=(10,10), reye=(10,30))
    self.assertEqual(bb.top, -4)
    self.assertEqual(bb.bottom, 43)
    self.assertEqual(bb.left, 0)
    self.assertEqual(bb.right, 39)

    # test that left and right profile versions work
    lbb = fd.utils.bounding_box_from_annotation(source='left-profile', mouth=(40,10), eye=(20,10))
    self.assertEqual(bb.top, -4)
    self.assertEqual(bb.bottom, 43)
    self.assertEqual(bb.left, 0)
    self.assertEqual(bb.right, 39)

    # test the direct way
    bb1 = fd.utils.bounding_box_from_annotation(topleft=(10,20), bottomright=(29,39))
    self.assertEqual(bb1.top, 10)
    self.assertEqual(bb1.bottom, 29)
    self.assertEqual(bb1.left, 20)
    self.assertEqual(bb1.right, 39)
    self.assertEqual(bb1.area, 400)

    bb2 = fd.utils.bounding_box_from_annotation(topleft=(15,25), bottomright=(34,44))
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


  def test03_mirror(self):
    bb = fd.utils.bounding_box_from_annotation(topleft=(10,20), bottomright=(29,39))
    mirrored = bb.mirror_x(60)
    self.assertEqual(mirrored.top, bb.top)
    self.assertEqual(mirrored.bottom, bb.bottom)
    self.assertEqual(mirrored.left, 20)
    self.assertEqual(mirrored.right, 39)

    # test that this IS actually, what we want
    image = numpy.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
    mirrored_image = image[:,::-1]
    bb = fd.BoundingBox(1,1,2,2)
    mb = bb.mirror_x(image.shape[1])
    x = image[bb.top:bb.bottom+1, bb.left:bb.right+1]
    y = mirrored_image[mb.top:mb.bottom+1, mb.left:mb.right+1]
    self.assertTrue((x == y[:,::-1]).all())


  def test10_pruning(self):
    # tests that the pruning functionality from C++ and python works similarly

    # read boxes and according detection values from files
    predictions = bob.io.load(bob.test.utils.datafile("detections.hdf5", 'xfacereclib.extension.facedetect', 'tests'))
    boxes = bob.io.load(bob.test.utils.datafile("boxes.hdf5", 'xfacereclib.extension.facedetect', 'tests'))
    detections = [fd.BoundingBox(boxes[i,0], boxes[i,1], boxes[i,2], boxes[i,3]) for i in range(boxes.shape[0])]

    # prune detections in the same way with C++ and python
    py_bb, py_val = fd.utils.boundingbox.prune(detections, predictions, 0.3)
    cpp_bb, cpp_val = fd.prune_detections(detections, predictions, 0.3)

    self.assertEqual(len(py_val), len(cpp_val))
    self.assertEqual(len(py_bb), 149)

    for (v1, v2, b1, b2) in zip(py_val, cpp_val, py_bb, cpp_bb):
      self.assertAlmostEqual(v1, v2)
      self.assertEqual(b1, b2)

