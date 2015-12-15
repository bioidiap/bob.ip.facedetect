import numpy
import os

import bob.ip.base
import bob.learn.boosting

import logging
logger = logging.getLogger('bob.ip.facedetect')

class Bootstrap:
  """This class deals with selecting new training examples for each boosting round.

  Bootstrapping a classifier works as follows:

  1. round = 1
  2. the classifier is trained on a random subset of the training data, where ``number_of_positive_examples_per_round`` and ``number_of_negative_examples_per_round`` defines the number of positive and negative examples used in the first round
  3. add ``number_of_weak_learners_in_first_round**round`` weak classifiers (selected using boosting)
  4. evaluate the whole training set using the current set of classifiers
  5. add the new data that is mis-classified by the largest margin to the training set
  6. round = round + 1
  7. if round < ``number_of_rounds`` goto 3

  **Constructor Documentation**

    Creates a new Bootstrap object that can be used to start or continue the training of a bootstrapped boosted classifier.

    **Parameters:**

    ``number_of_rounds`` : int
      The number of bootstrapping rounds, where each round adds more weak learners to the model

    ``number_of_weak_learners_in_first_round`` : int
      The number of weak classifiers chosen in the first round; later rounds will potentate this number, so don't choose it too large

    ``number_of_positive_examples_per_round, number_of_negative_examples_per_round`` : int
      The number of positive and negative samples added in each bootstrapping round; these numbers should be balanced, but do not necessarily need to be
  """
  def __init__(self, number_of_rounds = 7, number_of_weak_learners_in_first_round = 8, number_of_positive_examples_per_round = 5000, number_of_negative_examples_per_round = 5000):
    self.m_number_of_rounds = number_of_rounds
    self.m_number_of_weak_learners_per_round = [number_of_weak_learners_in_first_round * 2**i for i in range(number_of_rounds)]
    self.m_number_of_positive_examples_per_round = number_of_positive_examples_per_round
    self.m_number_of_negative_examples_per_round = number_of_negative_examples_per_round

  def run(self, training_set, trainer, filename = "bootstrapped_model.hdf5", force = False):
    """run(training_set, trainer, [filename], [force]) -> model

    Runs the bootstrapped training of a strong classifier using the given training data and a strong classifier trainer.
    The training set need to contain extracted features already, as this function will need the features several times.

    **Parameters:**

    ``training_set`` : :py:class:`TrainingSet`
      The training set containing pre-extracted feature files

    ``trainer`` : :py:class:`bob.learn.boosting.Boosting`
      A strong boosting trainer to use for selecting the weak classifiers and their weights for each round.

    ``filename`` : str
      A filename, where to write the resulting strong classifier to.
      This filename is also used as a base to compute filenames of intermediate files, which store results of each of the bootstrapping steps.

    ``force`` : bool
      If set to ``False`` (the default), the bootstrapping will continue the round, where it has been stopped during the last run (reading the current stage from respective files).
      If set to ``True``, the training will start from the beginning.

    **Returns:**

    ``model`` : :py:class:`bob.learn.boosting.BoostedMachine`
      The resulting strong classifier, a weighted combination of weak classifiers.
    """

    feature_extractor = training_set.feature_extractor()

    training_data = None
    training_labels = None
    model = None

    positive_indices, negative_indices = set(), set()

    for b in range(self.m_number_of_rounds):
      # check if old results are present
      temp_file = "%s_round_%d.hdf5" % (os.path.splitext(filename)[0], b+1)
      if os.path.exists(temp_file) and not force:
        logger.info("Loading already computed stage %d from %s.", b+1, temp_file)
        model, positives, negatives = self._load(bob.io.base.HDF5File(temp_file))
        positive_indices |= positives
        negative_indices |= negatives

      else:
        if positive_indices or negative_indices:
          # load data from previous rounds
          logger.info("Getting training data of previous rounds")
          training_data, training_labels = training_set.sample(positive_indices = positive_indices, negative_indices = negative_indices)
          positive_indices, negative_indices = set(), set()

        # get data for current round
        logger.info("Getting new data for bootstrapping round %d", b+1)
        new_data, new_labels = training_set.sample(model, self.m_number_of_positive_examples_per_round, self.m_number_of_negative_examples_per_round)
        if training_data is None:
          training_data = new_data
        else:
          training_data = numpy.append(training_data, new_data, axis=0)
        if training_labels is None:
          training_labels = new_labels
        else:
          training_labels = numpy.append(training_labels, new_labels, axis=0)

        logger.info("Starting training with %d examples", training_data.shape[0])
        model = trainer.train(training_data, training_labels, self.m_number_of_weak_learners_per_round[b], model)

        # write model and extractor to temporary file to be able to catch up later
        logger.info("Saving results for stage %d to file %s", b+1, temp_file)
        self._save(bob.io.base.HDF5File(temp_file, 'w'), model, training_set.positive_indices, training_set.negative_indices)

      feature_extractor.model_indices = model.indices

    # finally, return the trained model
    return model


  def _save(self, hdf5, model, positives, negatives):
    """Saves the given intermediate state of the bootstrapping to file."""
    # write the model and the training set indices to the given HDF5 file
    hdf5.set("PositiveIndices", sorted(list(positives)))
    hdf5.set("NegativeIndices", sorted(list(negatives)))
    hdf5.create_group("Model")
    hdf5.cd("Model")
    model.save(hdf5)
    del hdf5


  def _load(self, hdf5):
    """Loads the intermediate state of the bootstrapping from file."""
    positives = set(hdf5.get("PositiveIndices"))
    negatives = set(hdf5.get("NegativeIndices"))
    hdf5.cd("Model")
    model = bob.learn.boosting.BoostedMachine(hdf5)
    return model, positives, negatives
