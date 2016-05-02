import subprocess
import os
import shutil
import sys

import bob.extension
import bob.io.base.test_utils
list_file = bob.io.base.test_utils.temporary_filename(prefix="bobtest_", suffix='.txt')
cascade_file = bob.io.base.test_utils.temporary_filename(prefix="bobtest_", suffix='.hdf5')
detected_file = bob.io.base.test_utils.temporary_filename(prefix="bobtest_", suffix='.png')
import tempfile
feature_dir = tempfile.mkdtemp(prefix="bobtest_")

import bob.io.base
import bob.io.image
import bob.ip.color
import numpy

import bob.ip.facedetect


def test_collection():
  # Tests that the bin/collect_training_data.py works as expected

  executables = bob.extension.find_executable('collect_training_data.py', prefixes = [os.path.dirname(sys.argv[0]), 'bin'])
  assert executables
  executable = executables[0]

  assert not os.path.exists(list_file)

  command = [executable,
             '--image-directory', bob.io.base.test_utils.datafile("data", "bob.ip.facedetect", '.'),
             '--image-extension', '.jpg',
             '--annotation-directory', bob.io.base.test_utils.datafile("data", "bob.ip.facedetect", '.'),
             '--annotation-type', 'named',
             '--output-file', list_file,
             '-vvv'
            ]

  devnull = open(os.devnull, 'w')
  subprocess.call(command, stdout=devnull)
  assert os.path.exists(list_file)

  # check that we can read this file with our IO function
  train_set = bob.ip.facedetect.TrainingSet()
  train_set.load(list_file)

  data = list(train_set.iterate())
  assert len(data) == 1
  assert len(data[0]) == 3
  image, bboxes, filename = data[0]
  assert os.path.exists(filename)
  assert isinstance(image, numpy.ndarray)
  assert numpy.allclose(image, bob.ip.color.rgb_to_gray(bob.io.base.load(filename)))
  assert isinstance(bboxes, list)
  assert len(bboxes) == 1
  assert isinstance(bboxes[0], bob.ip.facedetect.BoundingBox)
  assert bboxes[0].similarity(bob.ip.facedetect.BoundingBox(topleft=(109.16, 85.30),size=(218.88, 182.4))) > 0.99


def test_extraction():
  # Tests that the bin/extract_training_features.py works as expected

  # check that the test before ran and produced the output file
  assert os.path.exists(list_file)
  try:
    executables = bob.extension.find_executable('extract_training_features.py', prefixes = [os.path.dirname(sys.argv[0]), 'bin'])
    assert executables
    executable = executables[0]

    # create command, as it was used for the default cascade
    command = [executable,
               '--file-lists', list_file,
               '--feature-directory', feature_dir,
               '--lbp-scale', '1',
               '--lbp-variant', 'mct',
               '-vvv'
              ]

    devnull = open(os.devnull, 'w')
    subprocess.call(command, stdout=devnull)

    assert os.path.exists(os.path.join(feature_dir, "Extractor.hdf5"))
    assert os.path.exists(os.path.join(feature_dir, "Features_00.hdf5"))

    # assert that we can load the extractor
    bob.ip.facedetect.FeatureExtractor(bob.io.base.HDF5File(os.path.join(feature_dir, "Extractor.hdf5")))

    # read the extracted features
    hdf5 = bob.io.base.HDF5File(os.path.join(feature_dir, "Features_00.hdf5"))
    assert hdf5.has_key("TotalPositives")
    assert hdf5.has_key("TotalNegatives")

    assert hdf5.get("TotalPositives") == 9 # yes, an odd number makes sense, even if we have mirrored images. The Sampler is the reason
    assert hdf5.get("TotalNegatives") == 1262

    assert hdf5.has_group("Image-0")
    hdf5.cd("Image-0")
    for k in hdf5.keys(relative=True):
      assert k.startswith("Positives") or k.startswith("Negatives")

  finally:
    # make sure that we delete the list file at the end, no matter what the test results in
    os.remove(list_file)


def test_training():
  # Tests that the bin/train_detector.py works as expected
  try:
    assert os.path.exists(os.path.join(feature_dir, "Extractor.hdf5"))
    assert os.path.exists(os.path.join(feature_dir, "Features_00.hdf5"))
    # we here emulate parallel execution by setting feature file to be the one of the first parallel job
    os.rename(os.path.join(feature_dir, "Features_00.hdf5"), os.path.join(feature_dir, "Features_01.hdf5"))

    executables = bob.extension.find_executable('train_detector.py', prefixes = [os.path.dirname(sys.argv[0]), 'bin'])
    assert executables
    executable = executables[0]

    command = [executable,
               '--feature-directory', feature_dir,
               '--trained-file', cascade_file,
               '--features-in-first-round', '2',
               '--bootstrapping-rounds', '2',
               '--classifiers-per-round', '3',
               '--training-examples', '5', '500',
               '-vvv'
              ]

    devnull = open(os.devnull, 'w')
    ret = subprocess.call(command, stdout=devnull)
    assert ret == 0

    # check that the training succeeded
    assert os.path.exists(cascade_file)
    # check that we can read the extracted cascade
    cascade = bob.ip.facedetect.Cascade(bob.io.base.HDF5File(cascade_file))

    assert numpy.allclose(cascade.thresholds, [-5,-5])
    assert len(cascade.cascade) == 2
    assert all(len(cascade.cascade[i].weak_machines) == 3 for i in (0,1))

  finally:
    # clean-up the mess of this and previous tests
    shutil.rmtree(feature_dir)


def test_detection():
  assert os.path.exists(cascade_file)
  # Tests the bin/extract_faces.py file
  try:
    executables = bob.extension.find_executable('detect_faces.py', prefixes = [os.path.dirname(sys.argv[0]), 'bin'])
    assert executables
    executable = executables[0]

    command = [executable,
               bob.io.base.test_utils.datafile("testimage.jpg", "bob.ip.facedetect"),
               '--cascade-file', cascade_file,
               '--lowest-scale', '0.5',
               '--prediction-threshold', '30',
               '--no-display',
               '--write-detection', detected_file,
               '-vvv'
              ]

    devnull = open(os.devnull, 'w')
    ret = subprocess.call(command, stdout=devnull)
    assert ret == 0

    # check that we can read the detected file
    assert os.path.exists(detected_file)
    image = bob.io.base.load(detected_file)

  finally:
    os.remove(cascade_file)
    if os.path.exists(detected_file):
      os.remove(detected_file)
