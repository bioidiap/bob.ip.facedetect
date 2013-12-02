import unittest
import math
from nose.plugins.skip import SkipTest

import bob
import numpy
import facereclib
import pkg_resources

import xfacereclib.extension.facedetect as fd
import xbob.db.verification.utils as vu

class SamplerTest (unittest.TestCase):

  def _make_boxes(self, detections):
    boxes = numpy.ndarray((len(detections), 4), numpy.int32)
    for i, (_, bb) in enumerate(detections):
      boxes[i,0] = bb.top
      boxes[i,1] = bb.left
      boxes[i,2] = bb.height
      boxes[i,3] = bb.width
    return boxes

  def test01_count(self):
    # test that the count of samples is correct
    test_image = bob.ip.rgb_to_gray(bob.io.load(bob.test.utils.datafile("testimage.jpg", 'xfacereclib.extension.facedetect', 'tests')))
    ground_truth = vu.read_annotation_file(bob.test.utils.datafile("testimage.pos", 'xfacereclib.extension.facedetect', 'tests'), 'named')

    # sample the image
    sampler = fd.detector.Sampler(distance=2, scale_factor=math.pow(2.,-1./4.), first_scale=0.5, cpp_implementation=True)
    sampler.add(test_image, [fd.utils.bounding_box_from_annotation(**ground_truth)])
    extractor = fd.FeatureExtractor(patch_size = (24,20), extractors = [bob.ip.LBP(8)])

    # extract test set features
    dataset, labels = sampler.get(extractor)

    self.assertEqual(numpy.count_nonzero(labels==1), 16)
    self.assertEqual(numpy.count_nonzero(labels==-1), 27617)

  def test02_detect(self):
    # test that the detection works as expected
    test_image = bob.ip.rgb_to_gray(bob.io.load(bob.test.utils.datafile("testimage.jpg", 'xfacereclib.extension.facedetect', 'tests')))

    classifier, extractor, is_cpp_extractor, mean, variance = fd.detector.load(bob.test.utils.datafile("extractor.hdf5", 'xfacereclib.extension.facedetect', 'tests'))
    sampler = fd.detector.Sampler(distance=2, scale_factor=math.pow(2.,-1./4.), first_scale=0.5, cpp_implementation=is_cpp_extractor)

    # get predictions
    feature = numpy.zeros(extractor.number_of_features, numpy.uint16)
    detections = [(classifier(feature), detection) for detection in sampler.iterate(test_image, extractor, feature)]
    self.assertEqual(len(detections), 28087)

    predictions = numpy.array([d[0] for d in detections])
    reference_file = bob.test.utils.datafile("detections.hdf5", 'xfacereclib.extension.facedetect', 'tests')

#    bob.io.save(predictions, reference_file)
    reference = bob.io.load(reference_file)
    for p,r in zip(predictions, reference):
      self.assertAlmostEqual(p, r)

    boxes = self._make_boxes(detections)
    reference_file = bob.test.utils.datafile("boxes.hdf5", 'xfacereclib.extension.facedetect', 'tests')
#    bob.io.save(boxes, reference_file)
    reference = bob.io.load(reference_file)
    self.assertEqual(numpy.count_nonzero(boxes != reference), 0)


  def test03_cascade(self):
    # test that the count of samples is correct
    test_image = bob.ip.rgb_to_gray(bob.io.load(bob.test.utils.datafile("testimage.jpg", 'xfacereclib.extension.facedetect', 'tests')))

    # sample the image
    classifier, extractor, is_cpp_extractor, mean, variance = fd.detector.load(bob.test.utils.datafile("extractor.hdf5", 'xfacereclib.extension.facedetect', 'tests'))
    sampler = fd.detector.Sampler(distance=2, scale_factor=math.pow(2.,-1./4.), first_scale=0.5, cpp_implementation=is_cpp_extractor)

    # get predictions
    feature = numpy.zeros(extractor.number_of_features, numpy.uint16)
    detections = list(sampler.cascade(test_image, extractor, feature, [classifier]))
    self.assertEqual(len(detections), 28087)

    # check value of a single prediction
    predictions = numpy.array([d[0] for d in detections])
    reference_file = bob.test.utils.datafile("detections.hdf5", 'xfacereclib.extension.facedetect', 'tests')

#    bob.io.save(predictions, reference_file)
    reference = bob.io.load(reference_file)
    for p,r in zip(predictions, reference):
      self.assertAlmostEqual(p, r)

    boxes = self._make_boxes(detections)
    reference_file = bob.test.utils.datafile("boxes.hdf5", 'xfacereclib.extension.facedetect', 'tests')
    reference = bob.io.load(reference_file)
    self.assertEqual(numpy.count_nonzero(boxes != reference), 0)

