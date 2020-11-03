import unittest
#from unittest import SkipTest
import random
import bob.learn.boosting
import numpy

class TestJesorskyLoss(unittest.TestCase):

  """ Test the loss function using multivariate data  """

  def test01_multivariate_dimensions(self):

    # Check the loss function values for multivariate targets

    loss_function = bob.learn.boosting.JesorskyLoss()
    num_samples = 2
    num_outputs = 4
    targets = numpy.array([[10, 10, 10, 30], [12, 11, 13, 29]], 'float64')
    score = numpy.array([[8, 9, 7, 34], [11, 6, 16, 26]], 'float64')
    alpha = numpy.array([0.5, 0.5, 0.5, 0.5])
    weak_scores = numpy.array([[0.2, 0.4, 0.5, 0.6], [0.5, 0.5, 0.5, 0.5]], 'float64')
    prev_scores = numpy.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.5, 0.5, 0.5]], 'float64')

    # check the loss dimensions
    loss_value = loss_function.loss(targets, score)
    self.assertTrue(loss_value.shape[0] == num_samples)
    self.assertTrue(loss_value.shape[1] == 1)

    # Check loss gradient
    grad_value = loss_function.loss_gradient( targets, score)
    self.assertTrue(grad_value.shape[0] == num_samples)
    self.assertTrue(grad_value.shape[1] == num_outputs)

    # Check loss sum
    loss_sum = loss_function.loss_sum(alpha, targets, prev_scores, weak_scores)
    self.assertTrue(loss_sum.shape[0] == 1)

    # Check the gradient sum
    grad_sum = loss_function.loss_gradient_sum(alpha, targets, prev_scores, weak_scores)
    self.assertTrue(grad_sum.shape[0] == num_outputs)


