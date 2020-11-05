from . import LossFunction

import numpy


class LogitLoss(LossFunction):
    """ The class to implement the logit loss function for the boosting framework."""

    def loss(self, targets, scores):
        """The function computes the logit loss values using prediction scores and targets.

        Keyword parameters:

          targets (float <#samples, #outputs>): The target values that should be reached.

          scores (float <#samples, #outputs>): The scores provided by the classifier.

        Returns
          (float <#samples, #outputs>): The loss values for the samples, which is always >= 0
        """
        e = numpy.exp(-(targets * scores))
        return numpy.log(1. + e)

    def loss_gradient(self, targets, scores):
        """The function computes the gradient of the logit loss function using prediction scores and targets.

        Keyword parameters:

          targets (float <#samples, #outputs>): The target values that should be reached.

          scores (float <#samples, #outputs>): The scores provided by the classifier.

        Returns
          loss (float <#samples, #outputs>): The gradient of the loss based on the given scores and targets.
        """
        e = numpy.exp(-(targets * scores))
        denom = 1. / (1. + e)
        return -targets * e * denom
