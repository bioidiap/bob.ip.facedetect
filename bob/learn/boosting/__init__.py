# import Libraries of other lib packages
import bob.extension
import bob.io.base
import bob.ip.facedetect
# versioning
from bob.learn.boosting import version
from bob.learn.boosting.version import module as __version__
from bob.learn.boosting.version import api as __api_version__

# include loss functions
from bob.learn.boosting.LossFunction import LossFunction # Just to get the documentation for it
from bob.learn.boosting.ExponentialLoss import ExponentialLoss
from bob.learn.boosting.LogitLoss import LogitLoss
from bob.learn.boosting.TangentialLoss import TangentialLoss
from bob.learn.boosting._library import JesorskyLoss

# include trainers
from bob.learn.boosting.StumpTrainer import StumpTrainer
from bob.learn.boosting.Boosting import Boosting
from bob.learn.boosting._library import LUTTrainer

# include machines
from bob.learn.boosting._library import WeakMachine, StumpMachine, LUTMachine, BoostedMachine

# include auxiliary functions
from bob.learn.boosting._library import weighted_histogram

def get_config():
  """Returns a string containing the configuration information.
  """
  return bob.extension.get_config(__name__, version.externals, version.api)


# gets sphinx autodoc done right - don't remove it
def __appropriate__(*args):
  """Says object was actually declared here, an not on the import module.
  Parameters:
    *args: An iterable of objects to modify
  Resolves `Sphinx referencing issues
  <https://github.com/sphinx-doc/sphinx/issues/3048>`
  """
  for obj in args: obj.__module__ = __name__
__appropriate__(
    LossFunction,
    )

# gets sphinx autodoc done right - don't remove it
__all__ = [_ for _ in dir() if not _.startswith('_')]

