import bob
import numpy
import facereclib


class FaceGraph:

  def __init__(self, patch_size = (96,80), gwt = bob.ip.GaborWaveletTransform(7), number_of_means = 25, subspace_size = None):
    self.gwt = gwt
    self.similarity = bob.machine.GaborJetSimilarity(bob.machine.gabor_jet_similarity_type.DISPARITY, gwt)
    self.patch_size = patch_size
    self.number_of_means = number_of_means
    self.subspace_size = subspace_size
    self.jet_image = self.gwt.empty_jet_image(numpy.ndarray(patch_size), True)

  # This algorithm is based on http://datasciencelab.wordpress.com/2013/12/12/clustering-with-k-means-in-python/
  def cluster(self, training_jets):
    jet_len = training_jets[0].shape[1]
    averager = bob.machine.GaborGraphMachine()

    # get initial centers randomly
    means = [training_jets[i] for i in facereclib.utils.quasi_random_indices(len(training_jets), self.number_of_means)]
    indices = {i:[] for i in range(self.number_of_means)}
    for r in range(100):
      facereclib.utils.debug("Starting K-Means round %d" % (r+1))

      # Assign all points in X to clusters
      clusters = {}
      last_indices = indices
      indices = {}
      for index, jet in enumerate(training_jets):
        similarities = [self.similarity(jet, mean) for mean in means]
        closest_mean_index = numpy.argmax(similarities)
        if closest_mean_index in clusters:
          clusters[closest_mean_index].append(jet)
          indices[closest_mean_index].append(index)
        else:
          clusters[closest_mean_index] = [jet]
          indices[closest_mean_index] = [index]

      # check for convergence
      converged = True
      for cluster_index in range(self.number_of_means):
        converged = converged and cluster_index in indices and cluster_index in last_indices and indices[cluster_index] == last_indices[cluster_index]
      if converged:
        facereclib.utils.info("K-Means converged after %d rounds" % (r+1))
        return means
      last_indices = indices

      # Re-evaluate centers -> create average jets
      old_means = means
      means = []
      for i in range(self.number_of_means):
        if i in clusters:
          graphs = numpy.expand_dims(numpy.array(clusters[i]), axis=1)
          average = numpy.ndarray([1, 2, jet_len])
          averager.average(graphs, average)
          means.append(average[0])
        else:
          # since we definitely need all clusters, we just re-initialize empty clusters with the training jet that is furthest from any mean
          # (should not happen often...)
          facereclib.utils.warn("Re-initialize K-Means index %d since the cluster is empty" % i)
          farest_training_sample = numpy.argmax([max(self.similarity(jet, mean) for mean in old_means) for jet in training_jets])
          means.append(training_jets[farest_training_sample])

    facereclib.utils.info("K-Means did not converge after 100 rounds; returning the latest means")
    return means



  def train(self, sampler, number_of_examples):
    """Extacts face graphs for the sampled patches."""
    # sample training images
    train_images, train_annotations = sampler.get_images_and_annotations(number_of_examples)

    number_of_nodes = train_annotations[0].shape[0]/2
    # correct training annotations since they currently are mean-free
    image_means = [self.patch_size[0]/2., self.patch_size[1]/2.] * number_of_nodes
    train_annotations = [t + image_means for t in train_annotations]

    # compute shape model
    facereclib.utils.info("Computing shape model from %d patches" % len(train_annotations))
    pca = bob.trainer.PCATrainer()
    self._subspace, _ = pca.train(train_annotations)
    if self.subspace_size is not None:
      self._subspace.resize(self._subspace.shape[0], self.subspace_size)

    facereclib.utils.info("Extracting graphs for %d patches" % len(train_images))
    graphs = []

    # extract face graphs
    for i in range(len(train_images)):
      self.gwt.compute_jets(train_images[i], self.jet_image, True)

      graph = []
      for a in range(0, train_annotations[i].shape[0], 2):
        y = (train_annotations[i][a]) % self.patch_size[0]
        x = (train_annotations[i][a+1]) % self.patch_size[1]

        graph.append(self.jet_image[y,x].copy())

      graphs.append(graph)

    if self.number_of_means is None:
      # just store the jets from ALL training patches
      self._graphs = numpy.array(graphs)
    else:
      # cluster the jets for each node
      position_count = self.number_of_predictions()/2
      self._graphs = numpy.zeros((self.number_of_means, position_count, 2, self.gwt.number_of_kernels))
      for i in range(position_count):
        facereclib.utils.info("Clustering jets for index %d" % i)
        jets = [graph[i] for graph in graphs]
        means = self.cluster(jets)
        for n in range(self.number_of_means):
          self._graphs[n,i] = means[n]



  def save(self, filename):
    f = bob.io.HDF5File(filename, 'w')
    f.set("Graphs", self._graphs)
    f.create_group("ShapeModel")
    f.create_group("Gabors")
    f.cd("ShapeModel")
    self._subspace.save(f)
    f.cd("../Gabors")
    self.gwt.save(f)


  def load(self, filename):
    f = bob.io.HDF5File(filename)
    self._graphs = f.read("Graphs")
    f.cd("ShapeModel")
    self._subspace = bob.machine.LinearMachine(f)
    f.cd("../Gabors")
    self.gwt = bob.ip.GaborWaveletTransform(f)

  def number_of_predictions(self):
    return self._subspace.input_subtract.shape[0]

  def optimize(self, jet_image, prediction):
    new_prediction = []
    # iterate over all positions
    for a in range(0, len(prediction), 2):
      # get the average position as a starting point
      y = prediction[a] % self.patch_size[0]
      x = prediction[a+1] % self.patch_size[1]

      # compute similarities and disparities for all stored graphs
      similarities = []
      disparities = []
      for graph in self._graphs:
        reference_jet = graph[a/2]
        target_jet = jet_image[y,x]
        similarities.append(self.similarity(reference_jet, target_jet))
        disparities.append(self.similarity.disparity())

      # get a sorted list of similarities and corresponding disparities
      best_similarity_indices = numpy.argsort(similarities)[:100]
#      similarities = [similarities[i] for i in best_similarity_indices]
      disparities = [disparities[i] for i in reversed(best_similarity_indices)]

      # histogram the directions of the disparities
      histogram = [[],[],[],[]]
      for disp in disparities:
        i = (2 if disp[0] < 0 else 0) + (1 if disp[1] < 1 else 0)
        histogram[i].append(disp)
      # check the most popular direction, and take the first element (which corresponds to the highest similarities)
      best_direction = numpy.argmax([len(h) for h in histogram])
      disparity = histogram[best_direction][0]


      # normalize similarities to become weights (ignore negative similarities)
#      similarities = numpy.array(similarities)
#      s = numpy.sum(similarities[similarities>0])
#      similarities = [sim/s for sim in similarities]
#      prediction.append(y + numpy.sum([similarities[i] * disparities[i][0] for i in range(len(similarities)) if similarities[i] > 0]))
#      prediction.append(x + numpy.sum([similarities[i] * disparities[i][1] for i in range(len(similarities)) if similarities[i] > 0]))

#      best_sim = numpy.argmax(similarities)
#      disparity = disparities[best_sim]

      new_prediction.append(y + disparity[0])
      new_prediction.append(x + disparity[1])

    return new_prediction


  def predict(self, patch):
    # perform gwt on image patch
    self.gwt.compute_jets(patch, self.jet_image, True)

    # compute prediction based on mean shape
    prediction = self.optimize(self.jet_image, self._subspace.input_subtract)

    # remove outliers with active shape model
    y = self._subspace.forward(numpy.array(prediction))
    prediction = numpy.inner(self._subspace.weights, y) + self._subspace.input_subtract

    """
    # predict again with new positions
    prediction = self.optimize(self.jet_image, prediction)
    """

    return prediction

