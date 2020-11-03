from ._library import LUTMachine, weighted_histogram

import numpy

class LUTTrainer():
  """The LUTTrainer contains methods to learn weak trainer using Look-Up-Tables (LUT).
  It can be used for uni-variate and multi-variate binary classification and, as well as for multi-variate regression.
  It requires that the features are discrete and have a maximum value.
  """

  def __init__(self, maximum_feature_value, feature_length, number_of_outputs = 1, selection_type = 'independent'):
    """Initializes the parameters of the LUT Trainer that trains a weak LUTMachine.

    Keyword parameters
      maximum_feature_value (int): The number of entries for the LUT (i.e., the maximum value of the discrete feature)

      feature_length (int): The length of the feature vectors

      number_of_outputs (int): The number of outputs for the multi-variate case.

      selection_type (str):
        The feature selection can be either 'independent' or 'shared'.
        For the independent case the loss function is separately considered for each output.
        For shared selection type the sum of the loss function is taken over the outputs and a single feature is used for all the outputs.
        See Cosmin's thesis for more details.

    """
    self.m_maximum_feature_value = maximum_feature_value
    self.m_feature_length = feature_length
    self.m_number_of_outputs = number_of_outputs
    self.m_selection_type = selection_type

    # pre-allocate arrays for faster access
    self._feature_gradient = numpy.ndarray((self.m_maximum_feature_value, self.m_number_of_outputs))
    self._luts = numpy.ndarray((self.m_maximum_feature_value, self.m_number_of_outputs))
    self._selected_indices = numpy.ndarray((self.m_number_of_outputs,), numpy.int32)
    self._gradient_histogram = numpy.ndarray((self.m_maximum_feature_value,))
    self._loss_sum = numpy.ndarray((self.m_feature_length, self.m_number_of_outputs))


  def train(self, training_features, loss_gradient):

    """Trains a weak LUTMachine.

    The function searches for a features index that minimizes (the length of) the loss gradient and computes the LUT corresponding to that feature index.

    Keyword parameters
      training_features (uint16<#samples, #features>): The training features samples

      loss_gradient (float<#samples, #outputs>): The loss gradient values for the training samples

    Returns
      A (weak) LUTMachine
    """

    # Compute the sum of the gradient based on the feature values or the loss associated with each feature index
    # Compute the loss for each feature
    for feature_index in range(self.m_feature_length):
      for output_index in range(self.m_number_of_outputs):
        weighted_histogram(training_features[:,feature_index], loss_gradient[:,output_index], self._gradient_histogram)
        self._loss_sum[feature_index, output_index] = - numpy.sum(numpy.abs(self._gradient_histogram))

    # Select the most discriminative index (or indices) for classification which minimizes the loss
    #  and compute the sum of gradient for that index
    if self.m_selection_type == 'independent':

      # independent feature selection is used if all the dimension of output use different feature
      # each of the selected feature minimize a dimension of the loss function
      for output_index in range(self.m_number_of_outputs):
        self._selected_indices[output_index] = self._loss_sum[:,output_index].argmin()

    else:  # shared

      # for 'shared' feature selection the loss function is summed over multiple dimensions and
      # the feature that minimized this cumulative loss is used for all the outputs
      accumulated_loss = numpy.sum(self._loss_sum,1)
      self._selected_indices.fill(accumulated_loss.argmin())

    for output_index in range(self.m_number_of_outputs):
      feature_index = self._selected_indices[output_index]
      self._feature_gradient[:,output_index] = weighted_histogram(training_features[:,feature_index], loss_gradient[:,output_index], self.m_maximum_feature_value)

    # Assign the values to LookUp Table
    self._luts.fill(1.)
    self._luts[self._feature_gradient <= 0.0] = -1.

    # create new weak machine
    return LUTMachine(self._luts.copy(), self._selected_indices.copy())

