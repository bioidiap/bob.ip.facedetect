import numpy
from .._library import FeatureExtractor

import bob.learn.boosting

class Cascade:

  """This class defines a cascade of strong classifiers :py:class:`bob.learn.boosting.BoostedMachine`.

  For each strong classifier, a threshold exists.
  When the weighted sum of predictions of classifiers gets below this threshold, the classification is stopped.

  **Constructor Documentation:**

    The constructor has two different ways to be called.
    The first and most obvious way is to load the cascade from the given ``cascade_file``.

    The second way instantiates an empty cascade, with the given ``feature_extractor``.
    Please use the :py:meth:`add` function to add new strong classifiers with according thresholds.

    **Parameters:**

    ``cascade_file`` : :py:class:`bob.io.base.HDF5File`
      An HDF5 file open for reading

    ``feature_extractor`` : :py:class:`FeatureExtractor`
      A feature extractor that will be used to extract features for the strong classifiers.
  """


  def __init__(self, cascade_file=None, feature_extractor=None):
    # initializes the cascade
    if cascade_file is not None:
      self.load(cascade_file)
    else:
      self.extractor = feature_extractor
      self.cascade = []
      self.indices = []
      self.thresholds = []

    self._indices()


  def add(self, classifier, threshold, begin=None, end=None):
    """Adds a new strong classifier with the given threshold to the cascade.

    **Parameters:**

    classifier : :py:class:`bob.learn.boosting.BoostedMachine`
      A strong classifier to add

    ``threshold`` : float
      The classification threshold for this cascade step

    ``begin``, ``end`` : int or ``None``
      If specified, only the weak machines with the indices ``range(begin,end)`` will be added.
    """
    boosted_machine = bob.learn.boosting.BoostedMachine()
    if begin is None: begin = 0
    if end is None: end = len(classifier.weak_machines)
    for i in range(begin, end):
      boosted_machine.add_weak_machine(classifier.weak_machines[i], classifier.weights[i])
    self.cascade.append(boosted_machine)
    self.thresholds.append(threshold)
    self._indices()


  def create_from_boosted_machine(self, boosted_machine, classifiers_per_round, classification_thresholds=-5.):
    """Creates this cascade from the given boosted machine, by simply splitting off strong classifiers that have classifiers_per_round weak classifiers.

    **Parameters:**

    ``boosted_machine`` : :py:class:`bob.learn.boosting.BoostedMachine`
      The strong classifier to split into a regular cascade.

    ``classifiers_per_round`` : int
      The number of classifiers that each cascade step should contain.

    ``classification_threshold`` : float
      A single threshold that will be applied in all rounds of the cascade.
    """
    indices = list(range(0, len(boosted_machine.weak_machines), classifiers_per_round))
    if indices[-1] != len(boosted_machine.weak_machines): indices.append(len(boosted_machine.weak_machines))
    self.cascade = []
    self.indices = []
    for i in range(len(indices)-1):
      machine = bob.learn.boosting.BoostedMachine()
      for index in range(indices[i], indices[i+1]):
        machine.add_weak_machine(boosted_machine.weak_machines[index], boosted_machine.weights[index, 0])
      self.cascade.append(machine)
    if isinstance(classification_thresholds, (int, float)):
      self.thresholds = [classification_thresholds] * len(self.cascade)
    else:
      self.thresholds = classification_thresholds


  def generate_boosted_machine(self):
    """generate_boosted_machine() -> strong

    Creates a single strong classifier from this cascade by concatenating all strong classifiers.

    **Returns:**

    ``strong`` : :py:class:`bob.learn.boosting.BoostedMachine`
      The strong classifier as a combination of all classifiers in this cascade.
    """
    strong = bob.learn.boosting.BoostedMachine()
    for machine, index in zip(self.cascade, self.indices):
      weak = machine.weak_machines
      weights = machine.weights
      for i in range(len(weak)):
        strong.add_weak_machine(weak[i], weights[i])

    return strong


  def _indices(self):
    # computes the list of indices from the current classifiers
    self.indices = []
    for classifier in self.cascade:
      self.indices.append(classifier.indices)
    self.feature = numpy.zeros(self.extractor.number_of_features, numpy.uint16)


  def prepare(self, image, scale):
    """Prepares the cascade for extracting features of the given image in the given scale.

    **Parameters:**

    ``image`` : array_like (2D, float)
      The image from which features will be extracted

    ``scale`` : float
      The scale of the image, for which features will be extracted
    """
    # prepare the feature extractor with the given image and scale
    self.extractor.prepare(image, scale)


  def __call__(self, bounding_box):
    """__call__(bounding_box) -> sum

    Computes the classification result of this cascade for the given bounding_box.

    The features will be extracted from the image at the scale that was set by the latest call to :py:meth:`prepare`.
    The classification result is obtained by summing all results of all cascade steps, as long as the sum is not below the threshold of the current cascade step.

    Finally, the sum is returned.

    **Parameters:**

    ``bounding_box`` : :py:class:`BoundingBox`
      The bounding box for which the features should be classified.
      Please assure that the bounding box is inside the image resolution at the scale that was set by the latest call to :py:meth:`prepare`.

    **Returns:**

    ``sum`` : float
      The sum of the cascaded classifiers (which might have been stopped before the last classifier)
    """

    # computes the classification for the given bounding box
    result = 0.
    for i in range(len(self.indices)):
      # extract the features that we need for this round
      self.extractor.extract_indexed(bounding_box, self.feature, self.indices[i])
      result += self.cascade[i](self.feature)
      if result < self.thresholds[i]:
        # break the cascade when the patch can already be rejected
        break
    return result


  def save(self, hdf5):
    """Saves this cascade into the given HDF5 file.

    **Parameters:**

    ``hdf5`` : :py:class:`bob.io.base.HDF5File`
      An HDF5 file open for writing
    """
    # write the cascade to file
    hdf5.set("Thresholds", self.thresholds)
    for i in range(len(self.cascade)):
      hdf5.create_group("Classifier_%d" % (i+1))
      hdf5.cd("Classifier_%d" % (i+1))
      self.cascade[i].save(hdf5)
      hdf5.cd("..")
    hdf5.create_group("FeatureExtractor")
    hdf5.cd("FeatureExtractor")
    self.extractor.save(hdf5)
    hdf5.cd("..")


  def load(self, hdf5):
    """Loads this cascade from the given HDF5 file.

    **Parameters:**

    ``hdf5`` : :py:class:`bob.io.base.HDF5File`
      An HDF5 file open for reading
    """
    # write the cascade to file
    self.thresholds = hdf5.read("Thresholds")
    self.cascade = []
    for i in range(len(self.thresholds)):
      hdf5.cd("Classifier_%d" % (i+1))
      self.cascade.append(bob.learn.boosting.BoostedMachine(hdf5))
      hdf5.cd("..")
    hdf5.cd("FeatureExtractor")
    self.extractor = FeatureExtractor(hdf5)
    hdf5.cd("..")
    self._indices()
