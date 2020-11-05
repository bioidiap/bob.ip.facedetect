import numpy


class LossFunction(object):
    """This is a base class for all loss functions implemented in pure python.
    It is simply a python re-implementation of the :py:class:`bob.learn.boosting.LossFunction` class.

    This class provides the interface for the L-BFGS optimizer.
    Please overwrite the loss() and loss_gradient() function (see below) in derived loss classes.
    """

    def __init__(self):
        pass

    def loss(self, targets, scores):
        """This function is to compute the loss for the given targets and scores.

        Keyword parameters:

          targets (float <#samples, #outputs>): The target values that should be reached.

          scores (float <#samples, #outputs>): The scores provided by the classifier.

        Returns
          (float <#samples, #outputs>) or (float <#samples, 1>): The loss based on the given scores and targets.
          Depending on the intended task, one of the two output variants should be chosen.
          For classification tasks, please use the former way (#samples, #outputs), while for regression tasks, use the latter (#samples, 1).
        """
        raise NotImplementedError("This is a pure abstract function. Please implement that in your derived class.")

    def loss_gradient(self, targets, scores):
        """This function is to compute the gradient of the loss for the given targets and scores.

        Keyword parameters:

          targets (float <#samples, #outputs>): The target values that should be reached.

          scores (float <#samples, #outputs>): The scores provided by the classifier.

        Returns
          loss (float <#samples, #outputs>): The gradient of the loss based on the given scores and targets.
        """
        raise NotImplementedError("This is a pure abstract function. Please implement that in your derived class.")

    def loss_sum(self, alpha, targets, previous_scores, current_scores):
        """The function computes the sum of the loss which is used to find the optimized values of alpha (x).

        The functions computes sum of loss values which is required during the line search step for the optimization of the alpha.
        This function is given as the input for the L-BFGS optimization function.

        Keyword parameters:

          alpha (float): The current value of the alpha.

          targets (float <#samples, #outputs>): The targets for the samples

          previous_scores (float <#samples, #outputs>): The cumulative prediction scores of the samples until the previous round of the boosting.

          current_scores (float <#samples, #outputs>): The prediction scores of the samples for the current round of the boosting.

        Returns

          (float <#outputs>) The sum of the loss values for the current value of the alpha
        """

        # compute the scores and loss for the current alpha
        scores = previous_scores + alpha * current_scores
        losses = self.loss(targets, scores)

        # compute the sum of the loss
        return numpy.sum(losses, 0)

    def loss_gradient_sum(self, alpha, targets, previous_scores, current_scores):
        """The function computes the gradient as the sum of the derivatives per sample which is used to find the optimized values of alpha.

        The functions computes sum of loss values which is required during the line search step for the optimization of the alpha.
        This function is given as the input for the L-BFGS optimization function.

        Keyword parameters:

          alpha (float): The current value of the alpha.

          targets (float <#samples, #outputs>): The targets for the samples

          previous_scores (float <#samples, #outputs>): The cumulative prediction scores of the samples until the previous round of the boosting.

          current_scores (float <#samples, #outputs>): The prediction scores of the samples for the current round of the boosting.

        Returns
          (float <#outputs>) The sum of the loss gradient for the current value of the alpha.
        """

        # compute the loss gradient for the updated score
        scores = previous_scores + alpha * current_scores
        loss_gradients = self.loss_gradient(targets, scores)

        # take the sum of the loss gradient values
        return numpy.sum(loss_gradients * current_scores, 0)
