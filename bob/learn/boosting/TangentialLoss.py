from . import LossFunction

import numpy


class TangentialLoss(LossFunction):
    """Tangent loss function, as described in http://www.svcl.ucsd.edu/projects/LossDesign/TangentBoost.html."""

    def loss(self, targets, scores):
        """The function computes the logit loss values using prediction scores and targets.

        Keyword parameters:

          targets (float <#samples, #outputs>): The target values that should be reached.

          scores (float <#samples, #outputs>): The scores provided by the classifier.

        Returns
          (float <#samples, #outputs>): The loss values for the samples, always >= 0
        """
        return (2. * numpy.arctan(targets * scores) - 1.) ** 2

    def loss_gradient(self, targets, scores):
        """The function computes the gradient of the tangential loss function using prediction scores and targets.

        Keyword parameters:

          targets (float <#samples, #outputs>): The target values that should be reached.

          scores (float <#samples, #outputs>): The scores provided by the classifier.

        Returns
          loss (float <#samples, #outputs>): The gradient of the loss based on the given scores and targets.
        """
        m = targets * scores
        numer = 4. * (2. * numpy.arctan(m) - 1.)
        denom = 1. + m ** 2
        return numer / denom
