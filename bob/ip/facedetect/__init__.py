# import Libraries of other lib packages
import bob.ip.base
import bob.io.base
import bob.learn.boosting

from . import version
from .version import module as __version__

from ._library import FeatureExtractor, BoundingBox, prune_detections, overlapping_detections
from .detector import Sampler, Cascade
from .train import TrainingSet, expected_eye_positions, bounding_box_from_annotation, read_annotation_file

from .detect import default_cascade, best_detection, detect_single_face, detect_all_faces


def get_config():
  """Returns a string containing the configuration information.
  """
  import bob.extension
  return bob.extension.get_config(__name__, version.externals)


# gets sphinx autodoc done right - don't remove it
__all__ = [_ for _ in dir() if not _.startswith('_')]
