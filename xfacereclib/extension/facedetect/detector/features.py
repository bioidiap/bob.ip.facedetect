
import bob
import numpy

class LBPFeatures:

  def __init__(self, patch_size = (24,20), model = None, lbp_extractors = None, overlap="ignored", square = False, **kwargs):
    """Generates the feature extractors possible for the given patch size."""
    if lbp_extractors is not None:
      self.m_lbps = lbp_extractors
    else:
      self.m_lbps = [bob.ip.LBP(8, float(ry), float(rx), **kwargs) for ry in range(1, patch_size[0] / 2) for rx in range(1, patch_size[1] / 2) if not square or ry == rx]

    self.m_patch_size = patch_size

    self._image = None
    self.init()
    self.set_model(model)


  def init(self):
    """Initializes shortcuts for faster feature extraction."""
    # pre-compute some shortcuts for faster access during feature extraction
    shapes = [lbp.get_lbp_shape(self.m_patch_size) for lbp in self.m_lbps]
    self._feature_patches = [numpy.ndarray(s, numpy.uint16) for s in shapes]
    feature_lengths = [i * j for (i,j) in shapes]
    self.number_of_features = sum(feature_lengths)

    self._feature_starts = [0]
    for length in feature_lengths:
      self._feature_starts.append(length + self._feature_starts[-1])

    # extract the shortcuts for the fast feature extraction using model information
    # the LUT stores: a triple of (index into the m_lbps, extracted pixel y, extracted pixel x)
    self._lut = [(lbp_index, y + self.m_lbps[lbp_index].offset[0], x + self.m_lbps[lbp_index].offset[1]) for lbp_index in range(len(self.m_lbps)) for y in range(shapes[lbp_index][0]) for x in range(shapes[lbp_index][1])]


  def set_model(self, model = None, feature_indices = None):
    """Sets the model to extract only specific features."""
    # set the model
    if model is None and feature_indices is None:
      # model is not specified, extract all features
      self._feature_indices = range(len(self._lut))
    elif feature_indices is None:
      # extract only the features used by the model
      self._feature_indices = model.feature_indices()
    else:
      self._feature_indices = feature_indices



  def maximum_label(self):
    """Returns the maximum LBP code."""
    return self.m_lbps[0].max_label


  def prepare(self, image):
    """Simply stores the image for later use."""
    self._image = image


  def _extract_image_patch(self, bb):
    """Returns the image patch for the given bounding box."""
    # get image patch
    return self._image[bb.top : bb.bottom+1, bb.left : bb.right+1]


  def extract(self, bounding_box, dataset, index):
    """Extracts the feature for the given bounding box and writes it directly to the given dataset at the given index."""
    patch = self._extract_image_patch(bounding_box)
    # simply extract all features
    for t, lbp in enumerate(self.m_lbps):
      # extract feature
      lbp(patch, self._feature_patches[t])
      dataset[index, self._feature_starts[t] : self._feature_starts[t+1]] = self._feature_patches[t].flatten()

  def _e(self, l, i, x, y):
    return l.extract(i, x, y)

  def extract_single(self, bounding_box, feature_vector):
    """Extracts and returns the feature vector for the given bounding box, respecting the model (if set)."""
    # extract only those features that need to be extracted
    for i in self._feature_indices:
      lut = self._lut[i]
#      feature_vector[i] = self.m_lbps[lut[0]].extract(self._image, lut[1] + bounding_box.m_top, lut[2] + bounding_box.m_left)
      feature_vector[i] = self._e(self.m_lbps[lut[0]],self._image, lut[1] + bounding_box.top, lut[2] + bounding_box.left)




class MBLBPFeatures (LBPFeatures):

  def __init__(self, patch_size = (24,20), model = None, lbp_extractors = None, overlap = False, square = False, **kwargs):
    """Generates the feature extractors possible for the given patch size."""
    if lbp_extractors is not None:
      self.m_lbps = lbp_extractors
    elif overlap:
      self.m_lbps = [bob.ip.LBP(8, (dy, dx), (dy-1, dx-1), **kwargs) for dy in range(1, patch_size[0] - 1) for dx in range(1, patch_size[1] - 1) if not square or dy == dx]
    else:
      self.m_lbps = [bob.ip.LBP(8, (dy, dx), **kwargs) for dy in range(1, patch_size[0] / 3 + 1) for dx in range(1, patch_size[1] / 3 + 1) if not square or dy == dx]

    self.m_patch_size = patch_size

    # integral image needs to be generated for each image
    # we here start with the one for the given patch size (for initialization purposes
    self._integral_image = numpy.ndarray((patch_size[0]+1,patch_size[1]+1), numpy.float64)

    # pre-compute some shortcuts for faster access during feature extraction
    shapes = [lbp.get_lbp_shape(self._integral_image, True) for lbp in self.m_lbps]
    self._feature_patches = [numpy.ndarray(s, numpy.uint16) for s in shapes]
    feature_lengths = [i * j for (i,j) in shapes]
    self.number_of_features = sum(feature_lengths)

    self._feature_starts = [0]
    for length in feature_lengths:
      self._feature_starts.append(length + self._feature_starts[-1])

    # extract the shortcuts for the fast feature extraction using model information
    # the LUT stores: a triple of (index into the m_lbps, extracted pixel y, extracted pixel x)
    self._lut = [(lbp_index, y + self.m_lbps[lbp_index].offset[0], x + self.m_lbps[lbp_index].offset[1]) for lbp_index in range(len(self.m_lbps)) for y in range(shapes[lbp_index][0]) for x in range(shapes[lbp_index][1])]

    self.set_model(model)


  def prepare(self, image):
    """Computes the integral image for the given image."""
    # compute the integral image for the given image
    self._integral_image.resize(image.shape[0]+1, image.shape[1]+1)
    bob.ip.integral(image, self._integral_image, True)


  def _extract_image_patch(self, bb):
    """Returns the image patch for the given bounding box."""
    # get image patch (+1 for the bottom/right index since we extract from integral image)
    return self._integral_image[bb.top : bb.bottom + 2, bb.left : bb.right + 2]


  def extract(self, bounding_box, dataset, index):
    """Extracts the feature for the given bounding box and writes it directly to the given dataset at the given index."""
    patch = self._extract_image_patch(bounding_box)
    # simply extract all features
    for t, lbp in enumerate(self.m_lbps):
      # extract feature
      lbp(patch, self._feature_patches[t], True)
      dataset[index, self._feature_starts[t] : self._feature_starts[t+1]] = self._feature_patches[t].flatten()


#  def _x(self, lbp, i, y, x):
#    return lbp.extract(i, y, x, True)

  def extract_single(self, bounding_box, feature_vector):
    """Extracts and returns the feature vector for the given bounding box, respecting the model (if set)."""
    # extract only those features that need to be extracted
    for i in self._feature_indices:
      lbp,y,x = self._lut[i]
      feature_vector[i] = self.m_lbps[lbp].extract(self._integral_image, y + bounding_box.top, x + bounding_box.left, True)
#      feature_vector[i] = self._x(self.m_lbps[lbp], self._integral_image, y + bounding_box.top, x + bounding_box.left)


def save(features, hdf5file):
  """Writes the given feature extractor to HDF5."""
  # write the class
  hdf5file.set_attribute("Class", features.__class__.__name__)
  hdf5file.set("PatchSize", features.m_patch_size)
  # write the LBP extractors
  for i in range(len(features.m_lbps)):
    hdf5file.create_group("LBP_%d"%(i+1))
    hdf5file.cd("LBP_%d"%(i+1))
    features.m_lbps[i].save(hdf5file)
    hdf5file.cd("..")


def load(hdf5file):
  """Reads the feature extractor from HDF5."""
  # read the class
  cls = hdf5file.get_attribute("Class")
  patch_size = hdf5file.read("PatchSize")
  patch_size = tuple([int(patch_size[i]) for i in range(2)])
  # read the LBP extractors
  lbps = []
  i = 1
  while hdf5file.has_group("LBP_%d"%(i)):
    hdf5file.cd("LBP_%d"%(i))
    lbps.append(bob.ip.LBP(hdf5file))
    hdf5file.cd("..")
    i += 1

  # create a new LBP extractor
  return {
    "LBPFeatures" : LBPFeatures,
    "MBLBPFeatures" : MBLBPFeatures
  } [cls](patch_size=patch_size, lbp_extractors = lbps)
