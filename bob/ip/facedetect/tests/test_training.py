import math
import nose
from nose.plugins.skip import SkipTest
import tempfile
import os, shutil

import numpy

import bob.io.base
import bob.io.base.test_utils
import bob.io.image
import bob.ip.base
import bob.ip.color


import bob.ip.facedetect as fd

regenerate_refs = False

def test_training_lists():
  # Tests that the IO of the training set works as expected

  # add sample image and annotation file to training set
  train_set = fd.train.TrainingSet()
  annotations = fd.train.read_annotation_file(bob.io.base.test_utils.datafile("testimage.pos", 'bob.ip.facedetect'), 'named')
  train_set.add_image(bob.io.base.test_utils.datafile("testimage.jpg", 'bob.ip.facedetect'), annotations)

  assert len(train_set) == 1

  # save to list file
  temp_file = tempfile.mkstemp(prefix="FD_", suffix=".txt")[1]
  train_set.save(temp_file)

  # load in another training set
  train_set_2 = fd.train.TrainingSet()
  train_set_2.load(temp_file)

  os.remove(temp_file)

  # assert that both lists contain similar content
  assert len(train_set) == len(train_set_2)
  i1, bb1, n1 = list(train_set.iterate())[0]
  i2, bb2, n2 = list(train_set_2.iterate())[0]

  assert (i1 == i2).all()
  assert n1 == n2
  assert len(bb1) == len(bb2)
  assert abs(bb1[0].top_f - bb2[0].top_f) < 1e-6
  assert abs(bb1[0].left_f - bb2[0].left_f) < 1e-6
  assert abs(bb1[0].bottom_f - bb2[0].bottom_f) < 1e-6
  assert abs(bb1[0].right_f - bb2[0].right_f) < 1e-6


def test_extraction():
  # Test that the count of training samples is correct

  temp_dir = tempfile.mkdtemp(prefix="FD_")

  try:
    # add sample image and annotation file to training set
    train_set = fd.train.TrainingSet(temp_dir)
    annotations = fd.train.read_annotation_file(bob.io.base.test_utils.datafile("testimage.pos", 'bob.ip.facedetect'), 'named')
    train_set.add_image(bob.io.base.test_utils.datafile("testimage.jpg", 'bob.ip.facedetect'), annotations)

    # create sampler and feature extractor
    sampler = fd.detector.Sampler(distance=2, scale_factor=math.pow(2.,-1./4.), lowest_scale=0.125)
    extractor = fd.FeatureExtractor(patch_size = (24,20), extractors = [bob.ip.base.LBP(8)])

    # extract features
    train_set.extract(sampler, extractor, number_of_examples_per_scale=(None, None), similarity_thresholds=(0.3,0.7))

    assert os.path.exists(os.path.join(temp_dir, "Extractor.hdf5"))
    assert os.path.exists(os.path.join(temp_dir, "Features_00.hdf5"))

    # get all features
    features, labels = train_set.sample()

    # assert that the number of features and labels is correct
    assert features.shape[0] == 14012
    lbp_shape = bob.ip.base.LBP(8).lbp_shape((24,20))
    assert features.shape[1] == lbp_shape[0] * lbp_shape[1]
    assert numpy.count_nonzero(labels==1) == 12
    assert numpy.count_nonzero(labels==-1) == 14000

    # check exact values
    if regenerate_refs:
      f = bob.io.base.HDF5File(bob.io.base.test_utils.datafile("training.hdf5", 'bob.ip.facedetect'), 'w')
      f.set("Features", features)
      f.set("Labels", labels)

    f = bob.io.base.HDF5File(bob.io.base.test_utils.datafile("training.hdf5", 'bob.ip.facedetect'))
    ref_features = f.get("Features")
    ref_labels = f.get("Labels")

    assert (features == ref_features).all()
    assert (labels == ref_labels).all()
  finally:
    if os.path.exists(temp_dir):
      shutil.rmtree(temp_dir)
