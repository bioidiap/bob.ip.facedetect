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

    # bounding box
    bb = xfacereclib.extension.facedetect.BoundingBox(10, 10, 24, 20)

    # three types of LBP features: simple, MB, OMB
    for i, lbp in enumerate((bob.ip.LBP(8, 2.), bob.ip.LBP(8, (3,3)), bob.ip.LBP(8, (3,3), (2,2)))):
      extractor = xfacereclib.extension.facedetect.FeatureExtractor(patch_size = (24,20), extractors = [lbp])

      feature_length = extractor.number_of_features

      # check that both feature extractors extract the same features
      test_image = bob.ip.rgb_to_gray(bob.io.load(bob.test.utils.datafile("testimage.jpg", 'facereclib', 'tests')))

      feature = numpy.ndarray((1,feature_length), dtype=numpy.uint16)

      extractor.prepare(test_image, 1)
      extractor(bb, feature, 0)

      # single features
      some = numpy.zeros(feature_length, dtype=numpy.uint16)

      indices = [20, 53, 66]
      extractor.model_indices = numpy.array(indices, numpy.int32)
      extractor(bb, some)

      for i in indices:
        self.assertEqual(some[i], feature[0,i])

      extractor.extract_indexed(bb, some, numpy.array(indices, numpy.int32))
      for i in indices:
        self.assertEqual(some[i], feature[0,i])


  def test02_multiple_LBP(self):
    # two bounding boxes (py and c++ version)
    bb = xfacereclib.extension.facedetect.BoundingBox(10, 10, 24, 20)

    for kwargs in ({'uniform':True}, {'rotation_invariant':True}, {'to_average':True, 'add_average_bit':True}, {'elbp_type':bob.ip.ELBPType.TRANSITIONAL}, {'elbp_type':bob.ip.ELBPType.DIRECTION_CODED}):
      for square in (True, False):
        extractors = [
            xfacereclib.extension.facedetect.FeatureExtractor(patch_size = (24,20), template=bob.ip.LBP(8, **kwargs), square=square),
            xfacereclib.extension.facedetect.FeatureExtractor(patch_size = (24,20), template=bob.ip.LBP(8, (1,1), **kwargs), square=square),
            xfacereclib.extension.facedetect.FeatureExtractor(patch_size = (24,20), template=bob.ip.LBP(8, (1,1), **kwargs), square=square, overlap=True)
        ]
        for extractor in extractors:
          # check that both have the same feature length
          feature_length = extractor.number_of_features

          # check that both feature extractors extract the same features
          test_image = bob.ip.rgb_to_gray(bob.io.load(bob.test.utils.datafile("testimage.jpg", 'facereclib', 'tests')))

          feature = numpy.ndarray((1,feature_length), dtype=numpy.uint16)

          extractor.prepare(test_image, 1)
          extractor(bb, feature, 0)

          # single features
          some = numpy.zeros(feature_length, dtype=numpy.uint16)

          indices = [20, 53, 66]
          extractor.model_indices = numpy.array(indices, numpy.int32)
          extractor(bb, some)

          for i in indices:
            self.assertEqual(some[i], feature[0,i])

          extractor.extract_indexed(bb, some, numpy.array(indices, numpy.int32))
          for i in indices:
            self.assertEqual(some[i], feature[0,i])

