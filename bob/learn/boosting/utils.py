# utilities to read the MNIST data

import tempfile
import tarfile
import os, shutil
import numpy
import bob.io.base
import bob.io.base.test_utils

class MNIST:
  def __init__(self):
    # loads the MNIST data from the packed data file
    tar = bob.io.base.test_utils.datafile("mnist.tar.bz2", __name__)
    temp_dir = tempfile.mkdtemp(prefix="bob_boosting_", suffix = "hdf5")

    f = tarfile.open(tar, 'r')
    f.extractall(temp_dir)
    del f

    datafile = os.path.join(temp_dir, "mnist_data.hdf5")
    assert os.path.exists(datafile)

    hdf5 = bob.io.base.HDF5File(datafile)
    self._data = {}
    self._labels = {}
    for group in ('train', 'test'):
      hdf5.cd(group)
      data = hdf5.read('data')
      labels = hdf5.read('labels')
      self._data[group] = data
      self._labels[group] = labels
      hdf5.cd('..')

    del hdf5
    shutil.rmtree(temp_dir)

  def data(self, groups = ('train', 'test'), labels=range(10)):
    """Returns the digits and the labels for the given labels"""
    if isinstance(groups, str):
      groups = (groups,)

    if isinstance(labels, int):
      labels = (labels,)

    _data = []
    _labels = []
    for group in groups:
      for i in range(self._labels[group].shape[0]):
        # check if the label is the desired one
        if self._labels[group][i] in labels:
          _data.append(self._data[group][i])
          _labels.append(self._labels[group][i])
    return numpy.array(_data, numpy.uint8), numpy.array(_labels, numpy.uint8)


