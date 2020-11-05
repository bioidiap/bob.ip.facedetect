from ._library import BoostedMachine
import numpy
import scipy.optimize
import logging
logger = logging.getLogger('bob.learn.boosting')


class Boosting:
  """ The class to boost the features from  a set of training samples.

  It iteratively adds new weak models to assemble a strong classifier.
  In each round of iteration a weak machine is learned by optimizing a differentiable function.

  **Constructor Documentation**

  Keyword parameters

    weak_trainer : :py:class:`bob.learn.boosting.LUTTrainer` or :py:class:`bob.learn.boosting.StumpTrainer`
      The class to train weak machines.

    loss_function : a class derived from :py:class:`bob.learn.boosting.LossFunction`
      The function to define the weights for the weak machines.

  """


  def __init__(self, weak_trainer, loss_function):
    self.m_trainer = weak_trainer
    self.m_loss_function = loss_function


  def get_loss_function(self):
    """Returns the loss function this trainer will use."""
    return self.m_loss_function


  def train(self, training_features, training_targets, number_of_rounds = 20, boosted_machine = None):
    """The function to train a boosting machine.

    The function boosts the training features and returns a strong classifier as a weighted combination of weak classifiers.

    Keyword parameters:

    training_features : uint16 <#samples, #features> or float <#samples, #features>)
      Features extracted from the training samples.

    training_targets : float <#samples, #outputs>
      The values that the boosted classifier should reach for the given samples.

    number_of_rounds : int
      The number of rounds of boosting, i.e., the number of weak classifiers to select.

    boosted_machine :py:class:`bob.learn.boosting.BoostedMachine` or None
      The machine to add the weak machines to. If not given, a new machine is created.

    Returns : :py:class:`bob.learn.boosting.BoostedMachine`
      The boosted machine that is combination of the weak classifiers.
    """

    # Initializations
    if(len(training_targets.shape) == 1):
      training_targets = training_targets[:,numpy.newaxis]

    number_of_samples = training_features.shape[0]
    number_of_outputs = training_targets.shape[1]

    strong_predicted_scores = numpy.zeros((number_of_samples, number_of_outputs))
    weak_predicted_scores = numpy.ndarray((number_of_samples, number_of_outputs))

    if boosted_machine is not None:
      boosted_machine(training_features, strong_predicted_scores)
    else:
      boosted_machine = BoostedMachine()

    # Start boosting iterations for num_rnds rounds
    logger.info("Starting %d rounds of boosting" % number_of_rounds)
    for round in range(number_of_rounds):

      logger.debug("Starting round %d" % (round+1))

      # Compute the gradient of the loss function, l'(y,f(x)) using loss_class
      loss_gradient = self.m_loss_function.loss_gradient(training_targets, strong_predicted_scores)

      # Select the best weak machine for current round of boosting
      weak_machine = self.m_trainer.train(training_features, loss_gradient)

      # Compute the classification scores of the samples based only on the current round weak classifier (g_r)
      weak_machine(training_features, weak_predicted_scores)

      # Perform L-BFGS minimization and compute the scale (alpha_r) for current weak machine
      alpha, _, flags = scipy.optimize.fmin_l_bfgs_b(
          func   = self.m_loss_function.loss_sum,
          x0     = numpy.zeros(number_of_outputs),
          fprime = self.m_loss_function.loss_gradient_sum,
          args   = (training_targets, strong_predicted_scores, weak_predicted_scores),
#          disp = 1
      )
      # check output of L-BFGS
      if flags['warnflag'] != 0:
        msg = "too many function evaluations or too many iterations" if flags['warnflag'] == 1 else flags['task']
        if (alpha == numpy.zeros(number_of_outputs)).all():
          logger.warn("L-BFGS returned zero weights with error '%d': %s" % (flags['warnflag'], msg))
          return boosted_machine
        else:
          logger.warn("L-BFGS returned warning '%d': %s" % (flags['warnflag'], msg))


      # Update the prediction score after adding the score from the current weak classifier f(x) = f(x) + alpha_r*g_r
      strong_predicted_scores += alpha * weak_predicted_scores

      # Add the current weak machine into the boosting machine
      boosted_machine.add_weak_machine(weak_machine, alpha)

      logger.info("Finished round %d / %d" % (round+1, number_of_rounds))

    return boosted_machine
