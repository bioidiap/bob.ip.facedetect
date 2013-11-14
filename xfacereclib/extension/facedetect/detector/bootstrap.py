
import numpy
import bob
import facereclib

class Bootstrap:
  """This class deals with selecting new training examples for each boosting round."""
  def __init__(self, number_of_rounds = 10, number_of_positive_examples_per_round = 10000, number_of_negative_examples_per_round = 10000):
    self.m_number_of_rounds = number_of_rounds
    self.m_number_of_positive_examples_per_round = number_of_positive_examples_per_round
    self.m_number_of_negative_examples_per_round = number_of_negative_examples_per_round

  def __call__(self, trainer, sampler, feature_extractor, model = None):
    training_data = numpy.ndarray((0,feature_extractor.number_of_features()), numpy.uint16)
    training_labels = numpy.ndarray((0,), numpy.float64)
    for b in range(self.m_number_of_rounds):
      # get new data
      facereclib.utils.info("Getting new data for bootstrapping round %d" % (b+1))
      new_data, new_labels = sampler.get(feature_extractor, model, self.m_number_of_positive_examples_per_round, self.m_number_of_negative_examples_per_round, True)
      training_data = numpy.append(training_data, new_data, axis=0)
      training_labels = numpy.append(training_labels, new_labels)

      facereclib.utils.info("Starting training with %d examples" % (training_data.shape[0]))
      model = trainer.train(training_data, training_labels, model)

      # write model to temporary file to be able to catch up later
      model.save(bob.io.HDF5File("Model_Round_%d.hdf5" % (b+1), 'w'))

      feature_extractor.set_model(model)

    # finally, return the trained model
    return model
