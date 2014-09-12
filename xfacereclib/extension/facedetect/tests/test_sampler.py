import unittest
import math
from nose.plugins.skip import SkipTest

import numpy
import facereclib
import pkg_resources

import bob.io.base
import bob.io.base.test_utils
import bob.io.image
import bob.ip.base
import bob.ip.color


import xfacereclib.extension.facedetect as fd
import bob.db.verification.utils as vu

regenerate_refs = False

class SamplerTest (unittest.TestCase):

  def _make_boxes(self, detections):
    boxes = numpy.ndarray((len(detections), 4), numpy.int32)
    for i, (_, bb) in enumerate(detections):
      boxes[i,0:2] = bb.topleft
      boxes[i,2:4] = bb.size
    return boxes

  def test01_count(self):
    # test that the count of samples is correct
    test_image = bob.ip.color.rgb_to_gray(bob.io.base.load(bob.io.base.test_utils.datafile("testimage.jpg", 'facereclib', 'tests')))
    ground_truth = vu.read_annotation_file(bob.io.base.test_utils.datafile("testimage.pos", 'facereclib', 'tests'), 'named')

    # sample the image
    sampler = fd.detector.Sampler(distance=2, scale_factor=math.pow(2.,-1./4.), lowest_scale=0.125)
    sampler.add(test_image, [fd.utils.bounding_box_from_annotation(**ground_truth)])
    extractor = fd.FeatureExtractor(patch_size = (24,20), extractors = [bob.ip.base.LBP(8)])

    # extract test set features
    dataset, labels = sampler.get(extractor)

    self.assertEqual(numpy.count_nonzero(labels==1), 12)
    self.assertEqual(numpy.count_nonzero(labels==-1), 14000)

  def test02_detect(self):
    # test that the detection works as expected
    test_image = bob.ip.color.rgb_to_gray(bob.io.base.load(bob.io.base.test_utils.datafile("testimage.jpg", 'facereclib', 'tests')))

    classifier, extractor, mean, variance = fd.detector.load(bob.io.base.test_utils.datafile("extractor.hdf5", 'xfacereclib.extension.facedetect', 'tests'))
    sampler = fd.detector.Sampler(distance=2, scale_factor=math.pow(2.,-1./4.), lowest_scale=0.125)

    # get predictions
    feature = numpy.zeros(extractor.number_of_features, numpy.uint16)
    detections = [(classifier(feature), detection) for detection in sampler.iterate(test_image, extractor, feature)]
    self.assertEqual(len(detections), 14493)

    predictions = numpy.array([d[0] for d in detections])
    reference_file = bob.io.base.test_utils.datafile("detections.hdf5", 'xfacereclib.extension.facedetect', 'tests')

    if regenerate_refs:
      bob.io.base.save(predictions, reference_file)
    reference = bob.io.base.load(reference_file)
    for p,r in zip(predictions, reference):
      self.assertAlmostEqual(p, r)

    boxes = self._make_boxes(detections)
    reference_file = bob.io.base.test_utils.datafile("boxes.hdf5", 'xfacereclib.extension.facedetect', 'tests')
    if regenerate_refs:
      bob.io.base.save(boxes, reference_file)
    reference = bob.io.base.load(reference_file)
    self.assertEqual(numpy.count_nonzero(boxes != reference), 0)


  def test03_cascade(self):
    # test that the count of samples is correct
    test_image = bob.ip.color.rgb_to_gray(bob.io.base.load(bob.io.base.test_utils.datafile("testimage.jpg", 'facereclib', 'tests')))

    # sample the image
    classifier, extractor, mean, variance = fd.detector.load(bob.io.base.test_utils.datafile("extractor.hdf5", 'xfacereclib.extension.facedetect', 'tests'))
    sampler = fd.detector.Sampler(distance=2, scale_factor=math.pow(2.,-1./4.), lowest_scale=0.125)

    # get predictions
    cascade = fd.detector.Cascade(classifier, classifiers_per_round=500, classification_thresholds=0., feature_extractor=extractor)
    detections = list(sampler.iterate_cascade(cascade, test_image))
    self.assertEqual(len(detections), 14493)

    # check value of a single prediction
    predictions = numpy.array([d[0] for d in detections])
    reference_file = bob.io.base.test_utils.datafile("detections.hdf5", 'xfacereclib.extension.facedetect', 'tests')

    if regenerate_refs:
      bob.io.base.save(predictions, reference_file)
    reference = bob.io.base.load(reference_file)
    for p,r in zip(predictions, reference):
      self.assertAlmostEqual(p, r)

    boxes = self._make_boxes(detections)
    reference_file = bob.io.base.test_utils.datafile("boxes.hdf5", 'xfacereclib.extension.facedetect', 'tests')
    reference = bob.io.base.load(reference_file)
    self.assertEqual(numpy.count_nonzero(boxes != reference), 0)

