from ._library import StumpMachine
import numpy

class StumpTrainer():
  """ The class for training weak stump classifiers.
  The weak stump is parameterized the threshold and the polarity.
  """

  def train(self, training_features, loss_gradient):
    """Computes a weak stump machine.

    The best weak machine is chosen to maximize the dot product of the outputs and the weights (gain).
    The weights are the negative of the loss gradient for exponential loss.

    Keyword parameters
      training_features (float<#samples, #features>): The training features samples

      loss_gradient (float<#samples>): The loss gradient values for the training samples

    Returns
      A (weak) :py:class:`bob.learn.boosting.StumpMachine`
    """

    # Initialization
    number_of_features = training_features.shape[1]
    threshold = numpy.zeros(number_of_features)
    polarity = numpy.zeros(number_of_features)
    gain = numpy.zeros(number_of_features)

    # For each feature find the optimum threshold, polarity and the gain
    for i in range(number_of_features):
      polarity[i], threshold[i], gain[i] = self.compute_threshold(training_features[:,i], -loss_gradient)

    #  Find the optimum id and its corresponding trainer
    best_index = gain.argmax()

    return StumpMachine(threshold[best_index], polarity[best_index], numpy.int32(best_index))


  def compute_threshold(self, features, gradient):
    """Computes the stump classifier threshold for a single feature

    The threshold is computed for the given feature values using the weak learner algorithm of Viola Jones.

    Keyword parameters
      features (float<#samples>): The feature values for a single index

      gradient (float<#samples>): The negative loss gradient values for the training samples

    Returns a triplet containing:
      threshold (float): threshold that minimizes the error
      polarity (float): the polarity or the direction used for stump classification
      gain (float): gain of the classifier
    """
    # Sort the feature and rearrange the corresponding weights and feature values
    sort_indices = numpy.argsort(features)
    features = features[sort_indices]
    gradient = gradient[sort_indices]

    unique_features, unique_indices = numpy.unique(features,return_index=True)

    if unique_features.shape[0] == 1:
      # if all features are identical, we gain nothing
      return 1., 0., 0.

    grad_cs = numpy.ndarray(unique_features.shape[0]-1)
    for i in range(1,unique_features.shape[0]):
      grad_cs[i-1] = numpy.sum(gradient[unique_indices[0]:unique_indices[i]])

    # For all the threshold compute the dot product
#    grad_cs =  numpy.cumsum(gradient)
    grad_sum = numpy.sum(gradient)
    gain = (grad_sum - grad_cs)

    # Find the index that maximizes the gain
    best_gain = numpy.argmax(numpy.absolute(gain))

    # Find the corresponding threshold value
    threshold = (unique_features[best_gain] + unique_features[best_gain+1])*0.5

    # Find the polarity or the directionality of the current trainer
    if gain[best_gain] > 0:
        polarity = -1
    else:
        polarity =  1

    # return polarity, threshold and the gain
    return polarity, threshold, abs(gain[best_gain])

