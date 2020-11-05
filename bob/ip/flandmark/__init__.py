# import Libraries of other lib packages
import bob.sp
import bob.ip.base
import bob.io.base
import bob.ip.facedetect
from ._library import *
from . import version
from .version import module as __version__

def get_config():
  """Returns a string containing the configuration information.
  """

  import bob.extension
  return bob.extension.get_config(__name__, version.externals)

def _init():
  # Setup default model for C-API
  from pkg_resources import resource_filename
  import os.path
  from ._library import _set_default_model
  _set_default_model(resource_filename(__name__, os.path.join('data', 'flandmark_model.dat')))

_init()

# gets sphinx autodoc done right - don't remove it
__all__ = [_ for _ in dir() if not _.startswith('_')]
