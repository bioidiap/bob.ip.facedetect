import unittest
import math
from nose.plugins.skip import SkipTest

import numpy
import facereclib
import pkg_resources

import bob.io.base
import bob.io.base.test_utils

import xfacereclib.extension.facedetect as fd

class UtilsTests(unittest.TestCase):

  def notest01_bounding_box_py(self):
    # Tests that the bounding box calculation works as expected

    # check the indirect ways using eye coordinates
    bb = fd.utils.bounding_box_from_annotation(leye=(10,10), reye=(10,30), padding={'left':-1, 'right':1, 'top':-1, 'bottom':1})
    self.assertEqual(bb.topleft, (-10, 0))
    self.assertEqual(bb.bottomright, (30,40))

    # check the scaling functionality
    sbb = bb.scale(0.5)
    self.assertEqual(sbb.topleft, (-5, 0))
    self.assertEqual(sbb.bottomright, (15,20))
    sbb = bb.scale(0.84)
    self.assertEqual(sbb.topleft, (-8, 0))
    self.assertEqual(sbb.bottomright, (26,34))


    bb = fd.utils.bounding_box_from_annotation(leye=(10,10), reye=(10,30))
    self.assertEqual(bb.top, -4)
    self.assertEqual(bb.bottom, 44)
    self.assertEqual(bb.left, 0)
    self.assertEqual(bb.right, 40)

    # test that left and right profile versions work
    lbb = fd.utils.bounding_box_from_annotation(source='left-profile', mouth=(40,10), eye=(20,10))
    self.assertEqual(bb.topleft, (-4,0))
    self.assertEqual(bb.bottomright, (44,40))

    # test the direct way
    bb1 = fd.utils.bounding_box_from_annotation(topleft=(10,20), bottomright=(30,40))
    self.assertEqual(bb1.topleft, (10,20))
    self.assertEqual(bb1.bottomright, (30,40))
    self.assertEqual(bb1.area, 400)

    bb2 = fd.utils.bounding_box_from_annotation(topleft=(15,25), bottomright=(35,45))
    bb3 = bb1.overlap(bb2)
    self.assertEqual(bb3.topleft, (15,25))
    self.assertEqual(bb3.bottomright, (30,40))
    self.assertEqual(bb3.area, 225)

    # check the similarity function
    self.assertEqual(bb1.similarity(bb1), 1.)
    self.assertEqual(bb3.similarity(bb2), 0.5625)
    self.assertEqual(bb3.similarity(bb1), bb1.similarity(bb3))


  def test02_bounding_box_cpp(self):
    # Tests that the bounding box calculation works as expected

    # check the indirect ways using eye coordinates
    bb = fd.utils.bounding_box_from_annotation(leye=(10,10), reye=(10,30), padding={'left':-1, 'right':1, 'top':-1, 'bottom':1})
    self.assertEqual(bb.topleft, (-10,0))
    self.assertEqual(bb.bottomright, (30,40))
    self.assertEqual(bb.size, (40,40))

    # check the scaling functionality
    sbb = bb.scale(0.5)
    self.assertEqual(sbb.topleft, (-5,0))
    self.assertEqual(sbb.bottomright, (15,20))
    sbb = bb.scale(2., centered=True)
    self.assertEqual(sbb.topleft, (-30,-20))
    self.assertEqual(sbb.bottomright, (50,60))
    sbb = bb.scale(0.84)
    self.assertEqual(sbb.topleft, (-8,0))
    self.assertEqual(sbb.bottomright, (25,34))


    bb = fd.utils.bounding_box_from_annotation(leye=(10,10), reye=(10,30))
    self.assertEqual(bb.topleft, (-4,0))
    self.assertEqual(bb.bottomright, (44,40))

    # test that left and right profile versions work
    lbb = fd.utils.bounding_box_from_annotation(source='left-profile', mouth=(40,10), eye=(20,10))
    self.assertEqual(bb.topleft, (-4,0))
    self.assertEqual(bb.bottomright, (44,40))

    # test the direct way
    bb1 = fd.utils.bounding_box_from_annotation(topleft=(10,20), bottomright=(30,40))
    self.assertEqual(bb1.topleft, (10,20))
    self.assertEqual(bb1.bottomright, (30,40))
    self.assertEqual(bb1.area, 400)

    bb2 = fd.utils.bounding_box_from_annotation(topleft=(15,25), bottomright=(35,45))
    bb3 = bb1.overlap(bb2)
    self.assertEqual(bb3.topleft, (15, 25))
    self.assertEqual(bb3.bottomright, (30,40))
    self.assertEqual(bb3.area, 225)

    # check the similarity function
    self.assertEqual(bb1.similarity(bb1), 1.)
    self.assertEqual(bb3.similarity(bb2), 0.5625)
    self.assertEqual(bb3.similarity(bb1), bb1.similarity(bb3))


  def test03_mirror(self):
    bb = fd.utils.bounding_box_from_annotation(topleft=(10,20), bottomright=(30,40))
    mirrored = bb.mirror_x(60)
    self.assertEqual(mirrored.topleft[0], bb.topleft[0])
    self.assertEqual(mirrored.bottomright[0], bb.bottomright[0])
    self.assertEqual(mirrored.topleft[1], 20)
    self.assertEqual(mirrored.bottomright[1], 40)

    # test that this IS actually, what we want
    image = numpy.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
    mirrored_image = image[:,::-1]
    bb = fd.BoundingBox((1,1),(2,2))
    mb = bb.mirror_x(image.shape[1])
    x = image[bb.topleft[0]:bb.bottomright[0], bb.topleft[1]:bb.bottomright[1]]
    y = mirrored_image[mb.topleft[0]:mb.bottomright[0], mb.topleft[1]:mb.bottomright[1]]
    self.assertTrue((x == y[:,::-1]).all())


  def test10_pruning(self):
    # tests that the pruning functionality from C++ and python works similarly

    # read boxes and according detection values from files
    predictions = bob.io.base.load(bob.io.base.test_utils.datafile("detections.hdf5", 'xfacereclib.extension.facedetect', 'tests'))
    boxes = bob.io.base.load(bob.io.base.test_utils.datafile("boxes.hdf5", 'xfacereclib.extension.facedetect', 'tests'))
    detections = [fd.BoundingBox(boxes[i,0:2], boxes[i,2:4]) for i in range(boxes.shape[0])]

    # prune detections in the same way with C++ and python
#    py_bb, py_val = fd.utils.boundingbox.prune(detections, predictions, 0.3)
    bb, val = fd.prune_detections(detections, predictions, 0.3)

#    self.assertEqual(len(py_val), len(cpp_val))
    self.assertEqual(len(bb), 149)
    self.assertEqual(len(val), 149)

