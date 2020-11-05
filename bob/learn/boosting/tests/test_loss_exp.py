import unittest
import random
import bob.learn.boosting
import numpy

class TestExponentialLoss(unittest.TestCase):
  """Perform test on exponential loss function """

  def test01_positive_target(self):
    # Loss values computation test for postitive targets.

    loss_function = bob.learn.boosting.ExponentialLoss()
    target = 1
    score = 0.34
    alpha = 0.5
    targets = numpy.array([1, 1, 1,1,1, 1,1,1,1,1])
    weak_scores = numpy.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], 'float64')
    prev_scores = numpy.array([0.53, 0.23, 0.63, 0.12, 1.34, 5.76, 3.21, 2.11, 1.21, 5.36], 'float64')

    # check the loss values
    loss_value = loss_function.loss(target, score)
    val = numpy.exp(- target * score)
    self.assertAlmostEqual(loss_value,val)
    self.assertTrue(loss_value >= 0)

    # Check loss gradient
    loss_grad = loss_function.loss_gradient( target, score)

    temp = numpy.exp(-target * score)
    val2 = -target * temp
    self.assertAlmostEqual(loss_grad,val2)

    # Check loss sum
    loss_sum_val = loss_function.loss_sum(alpha, targets, prev_scores, weak_scores)

    curr_scores = prev_scores + alpha*weak_scores
    val3 = sum(numpy.exp(-targets * curr_scores))
    self.assertAlmostEqual(val3, loss_sum_val)

    # Check the gradient sum
    grad_sum_val = loss_function.loss_gradient_sum(alpha, targets, prev_scores, weak_scores)

    curr_scores = prev_scores + alpha*weak_scores
    temp = numpy.exp(-targets * curr_scores)
    grad = -target * temp
    val4 = numpy.sum(grad * weak_scores,0)

    self.assertAlmostEqual(val4, grad_sum_val)


  def test02_negative_target(self):
    # Exponential Loss values computation test for negative targets.

    loss_function = bob.learn.boosting.ExponentialLoss()
    target = -1
    score = 0.34
    alpha = 0.5
    targets = numpy.array([-1, -1, -1,-1,-1, -1,-1,-1,-1,-1])
    weak_scores = numpy.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], 'float64')
    prev_scores = numpy.array([0.53, 0.23, 0.63, 0.12, 1.34, 5.76, 3.21, 2.11, 1.21, 5.36], 'float64')

    # check the loss values
    loss_value = loss_function.loss(target, score)
    val = numpy.exp(- target * score)
    self.assertAlmostEqual(loss_value,val)
    self.assertTrue(loss_value >= 0)

    # Check loss gradient
    loss_grad = loss_function.loss_gradient( target, score)

    temp = numpy.exp(-target * score)
    val2 = -target * temp
    self.assertAlmostEqual(loss_grad,val2)

    # Check loss sum
    loss_sum_val = loss_function.loss_sum(alpha, targets, prev_scores, weak_scores)

    curr_scores = prev_scores + alpha*weak_scores
    val3 = sum(numpy.exp(-targets * curr_scores))
    self.assertAlmostEqual(val3, loss_sum_val)

    # Check the gradient sum
    grad_sum_val = loss_function.loss_gradient_sum(alpha, targets, prev_scores, weak_scores)

    curr_scores = prev_scores + alpha*weak_scores
    temp = numpy.exp(-targets * curr_scores)
    grad = -target * temp
    val4 = numpy.sum(grad * weak_scores,0)

    self.assertAlmostEqual(val4, grad_sum_val)



  def test03_multivariate_dimensions(self):

    # Check the loss function values for multivariate targets

    loss_function = bob.learn.boosting.ExponentialLoss()
    num_samples = 3
    num_outputs = 2
    targets = numpy.array([[1, -1], [-1, 1], [0, 0]])
    score = numpy.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]], 'float64')
    alpha = 0.5
    weak_scores = numpy.array([[0.2, 0.4], [0.5, 0.6], [0.5, 0.5]], 'float64')
    prev_scores = numpy.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.5]], 'float64')

    # check the loss dimensions
    loss_value = loss_function.loss(targets, score)
    self.assertTrue(loss_value.shape[0] == num_samples)
    self.assertTrue(loss_value.shape[1] == num_outputs)

    # Check loss gradient
    grad_value = loss_function.loss_gradient( targets, score)
    self.assertTrue(grad_value.shape[0] == num_samples)
    self.assertTrue(grad_value.shape[1] == num_outputs)

    # Check loss sum
    loss_sum = loss_function.loss_sum(alpha, targets, prev_scores, weak_scores)
    self.assertTrue(loss_sum.shape[0] == num_outputs)


    # Check the gradient sum
    grad_sum = loss_function.loss_gradient_sum(alpha, targets, prev_scores, weak_scores)
    self.assertTrue(grad_sum.shape[0] == num_outputs)



  def test04_multivariate_negative_target(self):

    loss_function = bob.learn.boosting.ExponentialLoss()
    num_samples = 2
    num_dimension = 2
    targets = numpy.array([[1, -1], [-1, 1]])
    score = numpy.array([[0.5, 0.5], [0.5, 0.5]], 'float64')
    alpha = 0.5
    weak_scores = numpy.array([[0.2, 0.4], [0.5, 0.6]], 'float64')
    prev_scores = numpy.array([[0.1, 0.2],[0.3, 0.4]], 'float64')

    # check the loss values
    loss_value = loss_function.loss(targets, score)
    val1 = numpy.exp(- targets * score)
    self.assertTrue((loss_value == val1).all())

    # Check loss gradient
    loss_grad = loss_function.loss_gradient( targets, score)

    temp = numpy.exp(-targets * score)
    val2 = -targets * temp
    self.assertTrue((loss_grad == val2).all())

    # Check loss sum
    loss_sum_val = loss_function.loss_sum(alpha, targets, prev_scores, weak_scores)

    curr_scores = prev_scores + alpha*weak_scores
    val3 = sum(numpy.exp(-targets * curr_scores))
    self.assertTrue((val3 == loss_sum_val).all())

    # Check the gradient sum
    grad_sum_val = loss_function.loss_gradient_sum(alpha, targets, prev_scores, weak_scores)

    curr_scores = prev_scores + alpha*weak_scores
    temp = numpy.exp(-targets * curr_scores)
    grad = -targets * temp
    val4 = numpy.sum(grad * weak_scores,0)

    self.assertTrue((val4 == grad_sum_val).all())

