
import numpy
import bob
import facereclib
import os
from .io import save

class Bootstrap:
  """This class deals with selecting new training examples for each boosting round."""
  def __init__(self, number_of_rounds = 10, number_of_positive_examples_per_round = 10000, number_of_negative_examples_per_round = 10000):
    self.m_number_of_rounds = number_of_rounds
    self.m_number_of_positive_examples_per_round = number_of_positive_examples_per_round
    self.m_number_of_negative_examples_per_round = number_of_negative_examples_per_round

  def __call__(self, trainer, sampler, feature_extractor, model = None, filename="bootstrapped_model.hdf5"):
    training_data = numpy.ndarray((0,feature_extractor.number_of_features), numpy.uint16)
    training_labels = numpy.ndarray((0,), numpy.float64)
    mean = None
    variance = None
    for b in range(self.m_number_of_rounds):
      # get new data
      facereclib.utils.info("Getting new data for bootstrapping round %d" % (b+1))
      new_data, new_labels, new_means, new_variances = sampler.get(feature_extractor, model, self.m_number_of_positive_examples_per_round, self.m_number_of_negative_examples_per_round, delete_samples=True, compute_means_and_variances=True)
      if mean is None:
        mean = (min(new_means), max(new_means))
        variance = (min(new_variances), max(new_variances))
      elif new_means.size:
        mean = (min(min(new_means), mean[0]), max(max(new_means), mean[1]))
        variance = (min(min(new_variances), variance[0]), max(max(new_variances), variance[1]))
      training_data = numpy.append(training_data, new_data, axis=0)
      training_labels = numpy.append(training_labels, new_labels)

      facereclib.utils.info("Starting training with %d examples" % (training_data.shape[0]))
      model = trainer.train(training_data, training_labels, model)

      # write model and extractor to temporary file to be able to catch up later
      is_cpp = not hasattr(feature_extractor, "set_model")
      save("%s_round_%d.hdf5" % (os.path.splitext(filename)[0], b+1), model, feature_extractor, is_cpp, mean, variance)
      if is_cpp:
        feature_extractor.model_indices = model.indices
      else:
        feature_extractor.set_model(model)

    # finally, return the trained model
    return model, mean, variance
