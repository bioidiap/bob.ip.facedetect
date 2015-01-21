import numpy
import os

import bob.ip.base
import bob.learn.boosting

import logging
logger = logging.getLogger('bob.ip.facedetect')

class Bootstrap:
  """This class deals with selecting new training examples for each boosting round."""
  def __init__(self, number_of_rounds = 7, number_of_weak_learners_in_first_round = 8, number_of_positive_examples_per_round = 5000, number_of_negative_examples_per_round = 5000, force = False):
    self.m_number_of_rounds = number_of_rounds
    self.m_number_of_weak_learners_per_round = [number_of_weak_learners_in_first_round * 2**i for i in range(number_of_rounds)]
    self.m_number_of_positive_examples_per_round = number_of_positive_examples_per_round
    self.m_number_of_negative_examples_per_round = number_of_negative_examples_per_round

  def run(self, training_set, trainer, model = None, filename = "bootstrapped_model.hdf5", force = False):
    feature_extractor = training_set.feature_extractor()

    training_data = None
    training_labels = None

    positive_indices, negative_indices = set(), set()

    for b in range(self.m_number_of_rounds):
      # check if old results are present
      temp_file = "%s_round_%d.hdf5" % (os.path.splitext(filename)[0], b+1)
      if os.path.exists(temp_file) and not force:
        logger.info("Loading already computed stage %d from %s.", b+1, temp_file)
        model, positives, negatives = self.load(bob.io.base.HDF5File(temp_file))
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
        self.save(bob.io.base.HDF5File(temp_file, 'w'), model, training_set.positive_indices, training_set.negative_indices)

      feature_extractor.model_indices = model.indices

    # finally, return the trained model
    return model


  def save(self, hdf5, model, positives, negatives):
    # write the model and the training set indices to the given HDF5 file
    hdf5.set("PositiveIndices", sorted(list(positives)))
    hdf5.set("NegativeIndices", sorted(list(negatives)))
    hdf5.create_group("Model")
    hdf5.cd("Model")
    model.save(hdf5)
    del hdf5


  def load(self, hdf5):
    positives = set(hdf5.get("PositiveIndices"))
    negatives = set(hdf5.get("NegativeIndices"))
    hdf5.cd("Model")
    model = bob.learn.boosting.BoostedMachine(hdf5)
    return model, positives, negatives
