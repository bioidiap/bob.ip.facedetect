import numpy
from . import LossFunction


class ExponentialLoss(LossFunction):
    """ The class implements the exponential loss function for the boosting framework."""

    def loss(self, targets, scores):
        """The function computes the exponential loss values using prediction scores and targets.
        It can be used in classification tasks, e.g., in combination with the StumpTrainer.

        Keyword parameters:

          targets (float <#samples, #outputs>): The target values that should be reached.

          scores (float <#samples, #outputs>): The scores provided by the classifier.

        Returns
          (float <#samples, #outputs>): The loss values for the samples, always >= 0
        """
        return numpy.exp(-(targets * scores))

    def loss_gradient(self, targets, scores):
        """The function computes the gradient of the exponential loss function using prediction scores and targets.

        Keyword parameters:

          targets (float <#samples, #outputs>): The target values that should be reached.

          scores (float <#samples, #outputs>): The scores provided by the classifier.

        Returns
          loss (float <#samples, #outputs>): The gradient of the loss based on the given scores and targets.
        """
        loss = numpy.exp(-(targets * scores))
        return -targets * loss
