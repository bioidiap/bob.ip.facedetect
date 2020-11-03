import unittest
import random
import bob.learn.boosting
import numpy

class TestLogitLoss (unittest.TestCase):
  """Perform test on loss function """

  def test01_positive_target(self):
    # Check the loss function value for positive targets

    loss_function = bob.learn.boosting.LogitLoss()
    target = 1
    score = 0.34
    alpha = 0.5
    targets = numpy.array([1, 1, 1,1,1, 1,1,1,1,1])
    weak_scores = numpy.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], 'float64')
    prev_scores = numpy.array([0.53, 0.23, 0.63, 0.12, 1.34, 5.76, 3.21, 2.11, 1.21, 5.36], 'float64')

    # check the loss values
    loss_value = loss_function.loss(target, score)
    val1 = numpy.log(1 + numpy.exp(- target * score))
    self.assertAlmostEqual(loss_value,val1)

    # Check loss gradient
    grad_value = loss_function.loss_gradient( target, score)
    temp = numpy.exp(-target * score)
    val2 = -(target * temp* (1/(1 + temp)) )
    self.assertAlmostEqual(grad_value,val2)

    # Check loss sum
    loss_sum = loss_function.loss_sum(alpha, targets, prev_scores, weak_scores)
    curr_scores = prev_scores + alpha*weak_scores

    val3 = sum(numpy.log(1 + numpy.exp(-targets * curr_scores)))
    self.assertAlmostEqual(val3, loss_sum)

    # Check the gradient sum
    grad_sum = loss_function.loss_gradient_sum(alpha, targets, prev_scores, weak_scores)
    curr_scores = prev_scores + alpha*weak_scores
    temp = numpy.exp(-target * curr_scores)
    grad = -targets * temp *(1/ (1 + temp))
    val4 = numpy.sum(grad * weak_scores)
    self.assertAlmostEqual(val4, grad_sum)


  def test02_negative_target(self):
    # Check the loss function value for negative targets

    loss_function = bob.learn.boosting.LogitLoss()
    target = -1
    score = 0.34
    alpha = 0.5
    targets = numpy.array([-1, -1, -1,-1,-1, -1,-1,-1,-1,-1])
    weak_scores = numpy.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], 'float64')
    prev_scores = numpy.array([0.53, 0.23, 0.63, 0.12, 1.34, 5.76, 3.21, 2.11, 1.21, 5.36], 'float64')

    # check the loss values
    loss_value = loss_function.loss(target, score)
    val1 = numpy.log(1 + numpy.exp(- target * score))
    self.assertAlmostEqual(loss_value,val1)

    # Check loss gradient
    grad_value = loss_function.loss_gradient( target, score)
    temp = numpy.exp(-target * score)
    val2 = -(target * temp* (1/(1 + temp)) )
    self.assertAlmostEqual(grad_value,val2)

    # Check loss sum
    loss_sum = loss_function.loss_sum(alpha, targets, prev_scores, weak_scores)
    curr_scores = prev_scores + alpha*weak_scores

    val3 = sum(numpy.log(1 + numpy.exp(-targets * curr_scores)))
    self.assertAlmostEqual(val3, loss_sum)

    # Check the gradient sum
    grad_sum = loss_function.loss_gradient_sum(alpha, targets, prev_scores, weak_scores)
    curr_scores = prev_scores + alpha*weak_scores
    temp = numpy.exp(-target * curr_scores)
    grad = -targets * temp *(1/ (1 + temp))
    val4 = numpy.sum(grad * weak_scores)
    self.assertAlmostEqual(val4, grad_sum)


  def test03_multivariate_dimensions(self):
    # Check the loss function values for multivariate targets

    loss_function = bob.learn.boosting.LogitLoss()
    num_samples = 2
    num_dimension = 2
    targets = numpy.array([[1, -1], [-1, 1]])
    score = numpy.array([[0.5, 0.5], [0.5, 0.5]], 'float64')
    alpha = 0.5
    weak_scores = numpy.array([[0.2, 0.4], [0.5, 0.6]], 'float64')
    prev_scores = numpy.array([[0.1, 0.2],[0.3, 0.4]], 'float64')

    # check the loss dimensions
    loss_value = loss_function.loss(targets, score)
    self.assertTrue(loss_value.shape[0] == num_samples)
    self.assertTrue(loss_value.shape[1] == num_dimension)

    # Check loss gradient
    grad_value = loss_function.loss_gradient( targets, score)
    self.assertTrue(grad_value.shape[0] == num_samples)
    self.assertTrue(grad_value.shape[1] == num_dimension)

    # Check loss sum
    loss_sum = loss_function.loss_sum(alpha, targets, prev_scores, weak_scores)
    self.assertTrue(loss_sum.shape[0] == num_samples)

    # Check the gradient sum
    grad_sum = loss_function.loss_gradient_sum(alpha, targets, prev_scores, weak_scores)
    self.assertTrue(grad_sum.shape[0] == num_samples)



  def test04_multivariate(self):
    # Check the loss function values for multivariate targets

    loss_function = bob.learn.boosting.LogitLoss()
    targets = numpy.array([[1, -1], [-1, 1]])
    score = numpy.array([[0.5, 0.5], [0.5, 0.5]], 'float64')
    alpha = 0.5
    weak_scores = numpy.array([[0.2, 0.4], [0.5, 0.6]], 'float64')
    prev_scores = numpy.array([[0.1, 0.2],[0.3, 0.4]], 'float64')

    # check the loss values
    loss_value = loss_function.loss(targets, score)
    val1 = numpy.log(1 + numpy.exp(- targets * score))
    self.assertTrue((loss_value == val1).all())

    # Check loss gradient
    grad_value = loss_function.loss_gradient( targets, score)
    temp = numpy.exp(-targets * score)
    val2 = -(targets * temp* (1/(1 + temp)) )
    self.assertTrue((grad_value == val2).all())

    # Check loss sum
    loss_sum = loss_function.loss_sum(alpha, targets, prev_scores, weak_scores)
    curr_scores = prev_scores + alpha*weak_scores

    val3 = sum(numpy.log(1 + numpy.exp(-targets * curr_scores)))
    self.assertTrue((val3 == loss_sum).all())

    # Check the gradient sum
    grad_sum = loss_function.loss_gradient_sum(alpha, targets, prev_scores, weak_scores)
    curr_scores = prev_scores + alpha*weak_scores
    temp = numpy.exp(-targets * curr_scores)
    grad = -targets * temp *(1/ (1 + temp))
    val4 = sum(grad * weak_scores)
    self.assertTrue((val4 == grad_sum).all())
