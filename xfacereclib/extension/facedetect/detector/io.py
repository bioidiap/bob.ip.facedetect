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
    extractor.model_indices = model.feature_indices().astype(numpy.int64)
  else:
    extractor = load_features(f)
    extractor.set_model(model)
  del f

  return (model, extractor, is_cpp)


def save(filename, model, extractor, is_cpp=True):
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
  del f

