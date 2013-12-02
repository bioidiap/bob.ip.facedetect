import bob
import facereclib
import numpy
import os
import xbob.boosting
from .._features import FeatureExtractor
from .features import load as load_features
from .features import save as save_features

def load(filename):
  f = bob.io.HDF5File(filename)
  f.cd("/Machine")
  model = xbob.boosting.BoostedMachine(f)
  f.cd("/Features")
  is_cpp = not f.has_attribute("Class")
  if is_cpp:
    extractor = FeatureExtractor(f)
    extractor.model_indices = model.indices
  else:
    extractor = load_features(f)
    extractor.set_model(model)

  # check if mean and variance is stored
  f.cd("/")
  mean, variance = None, None
  if f.has_key('mean') and f.has_key('variance'):
    mean = f.read("mean")
    variance = f.read("variance")

  del f

  return (model, extractor, is_cpp, mean, variance)


def save(filename, model, extractor, is_cpp=True, mean=None, variance=None):
  facereclib.utils.ensure_dir(os.path.dirname(filename))
  f = bob.io.HDF5File(filename, 'w')
  f.create_group("Machine")
  f.create_group("Features")
  f.cd("/Machine")
  model.save(f)
  f.cd("/Features")
  if is_cpp:
    extractor.save(f)
  else:
    save_features(extractor, f)
  if mean is not None and variance is not None:
    f.cd("/")
    f.set("mean", numpy.array(mean))
    f.set("variance", numpy.array(variance))

  del f

