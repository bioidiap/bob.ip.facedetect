import facereclib
import numpy
import os
from .._library import FeatureExtractor

import bob.io.base
import bob.learn.boosting

def load(filename):
  f = bob.io.base.HDF5File(filename)
  f.cd("/Machine")
  model = bob.learn.boosting.BoostedMachine(f)
  f.cd("/Features")
  extractor = FeatureExtractor(f)
  extractor.model_indices = model.indices

  # check if mean and variance is stored
  f.cd("/")
  mean, variance = None, None
  if f.has_key('mean') and f.has_key('variance'):
    mean = f.read("mean")
    variance = f.read("variance")

  del f

  return (model, extractor, mean, variance)


def save(filename, model, extractor, mean=None, variance=None):
  facereclib.utils.ensure_dir(os.path.dirname(filename))
  f = bob.io.base.HDF5File(filename, 'w')
  f.create_group("Machine")
  f.create_group("Features")
  f.cd("/Machine")
  model.save(f)
  f.cd("/Features")
  extractor.save(f)
  if mean is not None and variance is not None:
    f.cd("/")
    f.set("mean", numpy.array(mean))
    f.set("variance", numpy.array(variance))

  del f

