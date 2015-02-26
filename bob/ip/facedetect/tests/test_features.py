import numpy

import bob.io.base
import bob.io.image
import bob.io.base.test_utils
import bob.ip.base
import bob.ip.color

import bob.ip.facedetect

def test01_single_LBP():
  # checks that the C++ implementation and the python implementation of the feature extractors give the same results

  # bounding box
  bb = bob.ip.facedetect.BoundingBox((10, 10), (24, 20))

  # three types of LBP features: simple, MB, OMB
  for i, lbp in enumerate((bob.ip.base.LBP(8, 2.), bob.ip.base.LBP(8, (3,3)), bob.ip.base.LBP(8, (3,3), (2,2)))):
    extractor = bob.ip.facedetect.FeatureExtractor(patch_size = (24,20), extractors = [lbp])

    assert len(extractor.extractors) == 1
    assert extractor.extractors[0] == lbp

    feature_length = extractor.number_of_features

    # check that both feature extractors extract the same features
    test_image = bob.ip.color.rgb_to_gray(bob.io.base.load(bob.io.base.test_utils.datafile("testimage.jpg", 'bob.ip.facedetect')))

    feature = numpy.ndarray((1,feature_length), dtype=numpy.uint16)

    extractor.prepare(test_image, 1)
    extractor.extract_all(bb, feature, 0)

    # single features
    some = numpy.zeros(feature_length, dtype=numpy.uint16)

    indices = [20, 53, 66]
    extractor.model_indices = numpy.array(indices, numpy.int32)
    extractor.extract_indexed(bb, some)

    for i in indices:
      assert some[i] == feature[0,i]

    extractor.extract_indexed(bb, some, numpy.array(indices, numpy.int32))
    for i in indices:
      assert some[i] == feature[0,i]


def test02_multiple_LBP():
  # two bounding boxes (py and c++ version)
  bb = bob.ip.facedetect.BoundingBox((10, 10), (24, 20))

  for kwargs in ({'uniform':True}, {'rotation_invariant':True}, {'to_average':True, 'add_average_bit':True}, {'elbp_type':"transitional"}, {'elbp_type':"direction-coded"}):
    for square in (True, False):
      extractors = [
          bob.ip.facedetect.FeatureExtractor(patch_size = (24,20), template=bob.ip.base.LBP(8, **kwargs), square=square),
          bob.ip.facedetect.FeatureExtractor(patch_size = (24,20), template=bob.ip.base.LBP(8, (1,1), **kwargs), square=square),
          bob.ip.facedetect.FeatureExtractor(patch_size = (24,20), template=bob.ip.base.LBP(8, (1,1), **kwargs), square=square, overlap=True)
      ]
      for extractor in extractors:
        # check that both have the same feature length
        feature_length = extractor.number_of_features

        # check that both feature extractors extract the same features
        test_image = bob.ip.color.rgb_to_gray(bob.io.base.load(bob.io.base.test_utils.datafile("testimage.jpg", 'bob.ip.facedetect')))

        feature = numpy.ndarray((1,feature_length), dtype=numpy.uint16)

        extractor.prepare(test_image, 1)
        extractor.extract_all(bb, feature, 0)

        # single features
        some = numpy.zeros(feature_length, dtype=numpy.uint16)

        indices = [20, 53, 66]
        extractor.model_indices = numpy.array(indices, numpy.int32)
        extractor.extract_indexed(bb, some)

        for i in indices:
          assert some[i] == feature[0,i]

        extractor.extract_indexed(bb, some, numpy.array(indices, numpy.int32))
        for i in indices:
          assert some[i] == feature[0,i]
