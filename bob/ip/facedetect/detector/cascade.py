import numpy
from .._features import FeatureExtractor

import bob.learn.boosting

class Cascade:

  def __init__(self, classifier=None, classifiers_per_round=None, classification_thresholds=None, feature_extractor=None, classifier_file=None):
    # initializes the cascade
    if classifier_file is not None:
      self.load(classifier_file)
    elif classifier is not None and classifiers_per_round is not None:
      # split the classifier and the feature extractor into cascades
      self.extractor = feature_extractor
      indices = range(0, len(classifier.weak_machines), classifiers_per_round)
      if indices[-1] != len(classifier.weak_machines): indices.append(len(classifier.weak_machines))
      self.cascade = []
      self.indices = []
      for i in range(len(indices)-1):
        machine = classifier.__class__()
        for index in range(indices[i], indices[i+1]):
          machine.add_weak_machine(classifier.weak_machines[index], classifier.weights[index, 0])
        self.cascade.append(machine)
      if isinstance(classification_thresholds, (int, float)):
        self.thresholds = [classification_thresholds] * len(self.cascade)
      else:
        self.thresholds = classification_thresholds

    else:
      self.extractor = feature_extractor
      self.cascade = []
      self.indices = []
      self.thresholds = []

    self._indices()

  def _indices(self):
    # computes the list of indices from the current classifiers
    self.indices = []
    for classifier in self.cascade:
      self.indices.append(classifier.indices)
    self.feature = numpy.zeros(self.extractor.number_of_features, numpy.uint16)


  def prepare(self, image, scale):
    # prepare the feature extractor with the given image and scale
    self.extractor.prepare(image, scale)

  def __call__(self, bounding_box):
    # computes the classification for the given bounding box
    result = 0.
    for i in range(len(self.indices)):
      self.extractor.extract_indexed(bounding_box, self.feature, self.indices[i])
      result += self.cascade[i](self.feature)
      if result < self.thresholds[i]:
        # break the cascade when the patch can already be rejected
        break

    return result


  def add(self, classifier, begin, end, threshold):
    self.cascade.append(classifier.__class__(classifier.weak_machines[begin:end], classifier.weights[begin:end, 0]))
    self.thresholds.append(threshold)
    self._indices()


  def save(self, hdf5):
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

