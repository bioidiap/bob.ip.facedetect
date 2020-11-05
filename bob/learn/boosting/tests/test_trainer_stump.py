import unittest
import random
import bob.learn.boosting
import numpy

class TestStumpTrainer(unittest.TestCase):
  """Perform test on stump weak trainer"""

  def test01_stump_limits(self):
    # test the stump trainer and check the basic limits on stump parameters
    trainer = bob.learn.boosting.StumpTrainer()
    rand_matrix = numpy.array([[-1.57248569,  0.92857928,  0.97908357, -0.0758847 , -0.34067902],
                   [ 0.88562798,  1.82759883, -0.55953264,  0.82822718,  2.29955421],
                   [ 1.03220648,  0.20467357,  0.67769647,  0.57652722,  0.45538562],
                   [ 1.49901643,  1.34450249,  0.08667704,  0.33658217, -1.32629319]], 'float64')

    n_samples = 4
    dim = 5
    x_train1 = rand_matrix + 4
    x_train2 = rand_matrix - 4
    x_train = numpy.vstack((x_train1, x_train2))
    y_train = numpy.hstack((numpy.ones(n_samples),-numpy.ones(n_samples)))

    scores = numpy.zeros(2*n_samples)
    t = y_train*scores
    loss = -y_train*(numpy.exp(y_train*scores))

    stump = trainer.train(x_train,loss)

    self.assertTrue(stump.threshold <= numpy.max(x_train))
    self.assertTrue(stump.threshold >= numpy.min(x_train))
    self.assertTrue(stump.feature_indices() >= 0)
    self.assertTrue(stump.feature_indices() < dim)


  def test02_stump_index(self):
    # test the stump trainer if the correct feature indices are selected
    trainer = bob.learn.boosting.StumpTrainer()
    rand_matrix = numpy.array([[-1.57248569,  0.92857928,  0.97908357, -0.0758847 , -0.34067902],
                   [ 0.88562798,  1.82759883, -0.55953264,  0.82822718,  2.29955421],
                   [ 1.03220648,  0.20467357,  0.67769647,  0.57652722,  0.45538562],
                   [ 1.49901643,  1.34450249,  0.08667704,  0.33658217, -1.32629319]], 'float64')


    num_samples = 4
    dim = 5
    selected_index = 2
    delta = 2
    x_train1 = rand_matrix + 0.1
    x_train2 = rand_matrix - 0.1
    x_train = numpy.vstack((x_train1, x_train2))
    x_train[0:num_samples,selected_index] = x_train[0:num_samples,selected_index] + delta
    x_train[num_samples+1:,selected_index] = x_train[num_samples +1:,selected_index] - delta
    y_train = numpy.hstack((numpy.ones(num_samples),-numpy.ones(num_samples)))

    scores = numpy.zeros(2*num_samples)
    loss = -y_train*(numpy.exp(y_train*scores))

    stump = trainer.train(x_train,loss)

    self.assertEqual(stump.feature_indices(), selected_index)


  def test03_stump_polarity(self):
    # test the stump trainer if the polarity is reversed with change in targets sign
    trainer = bob.learn.boosting.StumpTrainer()
    rand_matrix = numpy.array([[-1.57248569,  0.92857928,  0.97908357, -0.0758847 , -0.34067902],
                   [ 0.88562798,  1.82759883, -0.55953264,  0.82822718,  2.29955421],
                   [ 1.03220648,  0.20467357,  0.67769647,  0.57652722,  0.45538562],
                   [ 1.49901643,  1.34450249,  0.08667704,  0.33658217, -1.32629319]], 'float64')
    num_samples = 4
    dim = 5
    selected_index = 2
    delta = 2
    x_train1 = rand_matrix + 0.1
    x_train2 = rand_matrix - 0.1
    x_train = numpy.vstack((x_train1, x_train2))
    x_train[0:num_samples,selected_index] = x_train[0:num_samples,selected_index] + delta
    x_train[num_samples+1:,selected_index] = x_train[num_samples +1:,selected_index] - delta
    y_train = numpy.hstack((numpy.ones(num_samples),-numpy.ones(num_samples)))

    scores = numpy.zeros(2*num_samples)
    t = y_train*scores
    loss = -y_train*(numpy.exp(y_train*scores))

    stump = trainer.train(x_train,loss)

    self.assertEqual(stump.feature_indices(), selected_index)

    polarity = stump.polarity

    # test the check on polarity when the labels are reversed
    y_train = - y_train
    t = y_train*scores
    loss = -y_train*(numpy.exp(y_train*scores))

    stump = trainer.train(x_train,loss)
    polarity_rev = stump.polarity
    self.assertEqual(polarity, -polarity_rev)


  def test04_threshold(self):
    # test to check the threshold value of the weak trainer
    trainer = bob.learn.boosting.StumpTrainer()

    rand_matrix = numpy.array([[-1.57248569,  0.92857928,  0.97908357, -0.0758847 , -0.34067902],
                   [ 0.88562798,  1.82759883, -0.55953264,  0.82822718,  2.29955421],
                   [ 1.03220648,  0.20467357,  0.67769647,  0.57652722,  0.45538562],
                   [ 1.49901643,  1.34450249,  0.08667704,  0.33658217, -1.32629319]], 'float64')
    num_samples = 4
    dim = 5
    selected_index = 2
    x_train1 = rand_matrix + 0.1
    x_train2 = rand_matrix - 0.1
    delta1 = 4
    delta2 = 2
    x_train = numpy.vstack((x_train1, x_train2))
    x_train[0:num_samples,selected_index] = x_train[0:num_samples,selected_index] + delta1
    x_train[num_samples+1:,selected_index] = x_train[num_samples +1:,selected_index] + delta2
    y_train = numpy.hstack((numpy.ones(num_samples),-numpy.ones(num_samples)))

    scores = numpy.zeros(2*num_samples)
    loss = -y_train*(numpy.exp(y_train*scores))

    stump = trainer.train(x_train,loss)

    self.assertTrue(stump.threshold > delta2)
    self.assertTrue(stump.threshold < delta1)


  def test05_compute_thresh(self):
    # Test the threshold for a single feature
    trainer = bob.learn.boosting.StumpTrainer()

    num_samples = 10
    # The value of feature for class 1
    fea1 = 1
    # The value of the feature for class 2
    fea2 = 10

    # feature vector for 10 samples
    features = numpy.array([fea1, fea1,fea1,fea1,fea1,fea2,fea2,fea2,fea2,fea2])
    label = numpy.array([1,1,1,1,1,-1, -1, -1,-1,-1])

    scores = numpy.zeros(num_samples)
    loss = -label*(numpy.exp(label*scores))

    trained_polarity, trained_threshold, trained_gain = trainer.compute_threshold(features, loss)

    threshold = float(fea1 + fea2)/2
    self.assertEqual(trained_threshold, threshold)

    if(fea1 < fea2):
      polarity = -1
    else:
      polarity = 1

    self.assertEqual(trained_polarity, polarity)


  def test06_compute_thresh_rearrange(self):
    # test the threshold for single feature using a different permutation
    trainer = bob.learn.boosting.StumpTrainer()

    num_samples = 10
    # The value of feature for class 1
    fea1 = 1
    # The value of the feature for class 2
    fea2 = 10

    # feature vector for 10 samples
    features = numpy.array([fea1, fea1, fea2, fea1, fea2, fea1, fea2, fea1, fea2, fea2])
    label =   numpy.array([ 1,  1,   -1,   1,  -1,  1,   -1,   1,  -1,  -1])

    scores = numpy.zeros(num_samples)
    loss = -label*(numpy.exp(label*scores))

    trained_polarity, trained_threshold, trained_gain = trainer.compute_threshold(features, loss)

    threshold = float(fea1 + fea2)/2
    self.assertEqual(trained_threshold, threshold)

    if(fea1 < fea2):
      polarity = -1
    else:
      polarity = 1

    self.assertEqual(trained_polarity, polarity)


  def test07_compute_polarity(self):
    # test the polarity of the classifier
    trainer = bob.learn.boosting.StumpTrainer()

    num_samples = 10
    # The value of feature for class 1
    fea1 = 10
    # The value of the feature for class 2
    fea2 = 1

    # feature vector for 10 samples
    features = numpy.array([fea1, fea1, fea2, fea1, fea2, fea1, fea2, fea1, fea2, fea2])
    label =   numpy.array([ 1,  1,   -1,   1,  -1,  1,   -1,   1,  -1,  -1])

    scores = numpy.zeros(num_samples)
    loss = -label*(numpy.exp(label*scores))

    trained_polarity, trained_threshold, trained_gain = trainer.compute_threshold(features, loss)

    if(fea1 < fea2):
      polarity = -1
    else:
      polarity = 1

    self.assertEqual(trained_polarity, polarity)


