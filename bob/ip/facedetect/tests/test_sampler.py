import unittest
import math
from nose.plugins.skip import SkipTest

import numpy
import pkg_resources

import bob.io.base
import bob.io.base.test_utils
import bob.io.image
import bob.ip.base
import bob.ip.color


import bob.ip.facedetect as fd

regenerate_refs = False

def _make_boxes(detections):
  boxes = numpy.ndarray((len(detections), 4), numpy.int32)
  for i, (_, bb) in enumerate(detections):
    boxes[i,0:2] = bb.topleft
    boxes[i,2:4] = bb.size
  return boxes


def test_detection():
  # test that the detection works as expected
  test_image = bob.ip.color.rgb_to_gray(bob.io.base.load(bob.io.base.test_utils.datafile("testimage.jpg", 'bob.ip.facedetect')))

  cascade = fd.default_cascade()
  classifier = cascade.generate_boosted_machine()
  extractor = cascade.extractor
  extractor.model_indices = classifier.indices
  sampler = fd.detector.Sampler(distance=2, scale_factor=math.pow(2.,-1./4.), lowest_scale=0.125)

  # get predictions
  feature = numpy.zeros(extractor.number_of_features, numpy.uint16)
  detections = [(classifier(feature), detection) for detection in sampler.iterate(test_image, extractor, feature)]
  assert len(detections) == 14493

  predictions = numpy.array([d[0] for d in detections])

  # check that predictions are correct
  reference_file = bob.io.base.test_utils.datafile("detections.hdf5", 'bob.ip.facedetect')
  if regenerate_refs:
    bob.io.base.save(predictions, reference_file)

  reference = bob.io.base.load(reference_file)
  assert numpy.allclose(predictions, reference)

  # check that the extracted bounding boxes are correct
  boxes = _make_boxes(detections)
  reference_file = bob.io.base.test_utils.datafile("boxes.hdf5", 'bob.ip.facedetect')
  if regenerate_refs:
    bob.io.base.save(boxes, reference_file)
  reference = bob.io.base.load(reference_file)
  assert numpy.count_nonzero(boxes != reference) == 0


def test_cascade():
  # test that the cascade works as expected
  test_image = bob.ip.color.rgb_to_gray(bob.io.base.load(bob.io.base.test_utils.datafile("testimage.jpg", 'bob.ip.facedetect')))

  cascade = fd.default_cascade()

  # sample the image
  sampler = fd.detector.Sampler(distance=2, scale_factor=math.pow(2.,-1./4.), lowest_scale=0.125)

  # get predictions
  detections = list(sampler.iterate_cascade(cascade, test_image))
  assert len(detections) == 14493

  # check value of a single prediction
  predictions = numpy.array([d[0] for d in detections])
  reference_file = bob.io.base.test_utils.datafile("detections.hdf5", 'bob.ip.facedetect')

  if regenerate_refs:
    bob.io.base.save(predictions, reference_file)
  reference = bob.io.base.load(reference_file)
  # check that the high values are the same
  # (for prediction values < -5, the cascade might have been exitted early)
  assert numpy.allclose(predictions[predictions>-5], reference[predictions>-5])

  boxes = _make_boxes(detections)
  reference_file = bob.io.base.test_utils.datafile("boxes.hdf5", 'bob.ip.facedetect')
  reference = bob.io.base.load(reference_file)
  assert numpy.count_nonzero(boxes != reference) == 0
