import unittest
import math
from nose.plugins.skip import SkipTest

import bob
import numpy
import facereclib
import pkg_resources

import xfacereclib.extension.facedetect

class ExtractorTests(unittest.TestCase):

  def test01_single_LBP(self):
    # checks that the C++ implementation and the python implementation of the feature extractors give the same results

    # two bounding boxes (py and c++ version)
    py_bb = xfacereclib.extension.facedetect.utils.BoundingBox('direct', topleft=(10,10), bottomright=(33,29))
    cpp_bb = xfacereclib.extension.facedetect.BoundingBox(10, 10, 24, 20)
    # three types of LBP features: simple, MB, OMB
    for i, lbp in enumerate((bob.ip.LBP(8, 2.), bob.ip.LBP(8, (3,3)), bob.ip.LBP(8, (3,3), (2,2)))):
      if i:
        py = xfacereclib.extension.facedetect.detector.MBLBPFeatures(patch_size = (24,20), lbp_extractors = [lbp])
      else:
        py = xfacereclib.extension.facedetect.detector.LBPFeatures(patch_size = (24,20), lbp_extractors = [lbp])
      cpp = xfacereclib.extension.facedetect.FeatureExtractor(patch_size = (24,20), extractors = [lbp])

      # check that both have the same feature length
      feature_length = py.number_of_features
      self.assertEqual(cpp.number_of_features, feature_length)

      # check that both feature extractors extract the same features
      test_image = bob.ip.rgb_to_gray(bob.io.load(bob.test.utils.datafile("testimage.jpg", 'facereclib', 'tests')))

      py_feat = numpy.ndarray((1,feature_length), dtype=numpy.uint16)
      cpp_feat = numpy.ndarray((1,feature_length), dtype=numpy.uint16)

      py.prepare(test_image, 1)
      py.extract(py_bb, py_feat, 0)

      cpp.prepare(test_image, 1)
      cpp(cpp_bb, cpp_feat, 0)

      self.assertEqual(numpy.count_nonzero(py_feat - cpp_feat), 0)

      # single features
      py_some = numpy.zeros(feature_length, dtype=numpy.uint16)
      cpp_some = numpy.zeros(feature_length, dtype=numpy.uint16)

      indices = [20, 53, 66]
      py.set_model(feature_indices = indices)
      py.extract_single(py_bb, py_some)

      cpp.model_indices = numpy.array(indices, numpy.int32)
      cpp(cpp_bb, cpp_some)

      self.assertEqual(numpy.count_nonzero(py_some - cpp_some), 0)
      for i in indices:
        self.assertEqual(cpp_some[i], cpp_feat[0,i])

      cpp.extract_indexed(cpp_bb, cpp_some, numpy.array(indices, numpy.int32))
      self.assertEqual(numpy.count_nonzero(py_some - cpp_some), 0)
      for i in indices:
        self.assertEqual(cpp_some[i], cpp_feat[0,i])


  def test02_multiple_LBP(self):
    # two bounding boxes (py and c++ version)
    py_bb = xfacereclib.extension.facedetect.utils.BoundingBox('direct', topleft=(10,10), bottomright=(33,29))
    cpp_bb = xfacereclib.extension.facedetect.BoundingBox(10, 10, 24, 20)

    for kwargs in ({'uniform':True}, {'rotation_invariant':True}, {'to_average':True, 'add_average_bit':True}, {'elbp_type':bob.ip.ELBPType.TRANSITIONAL}, {'elbp_type':bob.ip.ELBPType.DIRECTION_CODED}):
      for square in (True, False):
        py_f = [
            xfacereclib.extension.facedetect.detector.LBPFeatures(patch_size = (24,20), square=square, **kwargs),
            xfacereclib.extension.facedetect.detector.MBLBPFeatures(patch_size = (24,20), square=square, **kwargs),
            xfacereclib.extension.facedetect.detector.MBLBPFeatures(patch_size = (24,20), overlap=True, square=square, **kwargs),
        ]
        cpp_f = [
            xfacereclib.extension.facedetect.FeatureExtractor(patch_size = (24,20), template=bob.ip.LBP(8, **kwargs), square=square),
            xfacereclib.extension.facedetect.FeatureExtractor(patch_size = (24,20), template=bob.ip.LBP(8, (1,1), **kwargs), square=square),
            xfacereclib.extension.facedetect.FeatureExtractor(patch_size = (24,20), template=bob.ip.LBP(8, (1,1), **kwargs), square=square, overlap=True)
        ]
        for py, cpp in zip(py_f, cpp_f):
          # check that both have the same feature length
          feature_length = py.number_of_features
          self.assertEqual(cpp.number_of_features, feature_length)

          # check that both feature extractors extract the same features
          test_image = bob.ip.rgb_to_gray(bob.io.load(bob.test.utils.datafile("testimage.jpg", 'facereclib', 'tests')))

          py_feat = numpy.ndarray((1,feature_length), dtype=numpy.uint16)
          cpp_feat = numpy.ndarray((1,feature_length), dtype=numpy.uint16)

          py.prepare(test_image, 1)
          py.extract(py_bb, py_feat, 0)

          cpp.prepare(test_image, 1)
          cpp(cpp_bb, cpp_feat, 0)

          self.assertEqual(numpy.count_nonzero(py_feat - cpp_feat), 0)

          # single features
          py_some = numpy.zeros(feature_length, dtype=numpy.uint16)
          cpp_some = numpy.zeros(feature_length, dtype=numpy.uint16)

          indices = [20, 53, 66]
          py.set_model(feature_indices = indices)
          py.extract_single(py_bb, py_some)

          cpp.model_indices = numpy.array(indices, numpy.int32)
          cpp(cpp_bb, cpp_some)

          self.assertEqual(numpy.count_nonzero(py_some - cpp_some), 0)
          for i in indices:
            self.assertEqual(cpp_some[i], cpp_feat[0,i])

          cpp.extract_indexed(cpp_bb, cpp_some, numpy.array(indices, numpy.int32))
          self.assertEqual(numpy.count_nonzero(py_some - cpp_some), 0)
          for i in indices:
            self.assertEqual(cpp_some[i], cpp_feat[0,i])

