import bob
import numpy
import facereclib
import sys

from .LocalModel import LocalModel


class FaceGraph (LocalModel):

  def __init__(self, patch_size = (96,80), gwt = bob.ip.GaborWaveletTransform(9), number_of_means = 25, subspace_size = 0.98, offset = (30,30)):
    LocalModel.__init__(self, patch_size, subspace_size, offset)

    self.gwt = gwt
    self.similarity = bob.machine.GaborJetSimilarity(bob.machine.gabor_jet_similarity_type.DISPARITY, gwt)
    self.number_of_means = number_of_means
    self.jet_image = self.gwt.empty_jet_image(numpy.ndarray([patch_size[i] + 2 * offset[i] for i in (0,1)]), True)


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
          # compute average from training jets
          graphs = numpy.expand_dims(numpy.array(clusters[i]), axis=1)
          average = numpy.ndarray([1, 2, jet_len])
          averager.average(graphs, average)
          mean = average[0]
          # compute phase shift towards average jet
          graphs = numpy.expand_dims(numpy.array([self.similarity.shift_phase(graphs[m,0,:,:], mean) for m in range(graphs.shape[0])]), axis=1)
          averager.average(graphs, average)
# doesn't improve
#          # now, find the jet that is closest to the mean
#          index = numpy.argmax([self.similarity(jet, average[0]) for jet in clusters[i]])
#          means.append(clusters[i][index])
          means.append(average[0])
        else:
          # since we definitely need all clusters, we just re-initialize empty clusters with the training jet that is furthest from any mean
          # (should not happen often...)
          facereclib.utils.warn("Re-initialize K-Means index %d since the cluster is empty" % i)
          farest_training_sample = numpy.argmin([min(self.similarity(jet, mean) for mean in old_means) for jet in training_jets])
          means.append(training_jets[farest_training_sample])

    facereclib.utils.info("K-Means did not converge after 100 rounds; returning the latest means")
    return means


  def train_local_model(self, train_images, train_annotations):
    """Extracts face graphs for the sampled patches."""

    facereclib.utils.info("Extracting graphs for %d patches" % len(train_images))
    graphs = []

    # extract face graphs
    for i in range(len(train_images)):
      sys.stdout.write("\rProcessing patch %d of %d" % (i+1, len(train_images)))
      sys.stdout.flush()
      try:
        self.gwt.compute_jets(train_images[i], self.jet_image, True)

        graph = []
        for a in range(0, train_annotations[i].shape[0], 2):
          y = train_annotations[i][a]
          x = train_annotations[i][a+1]

          graph.append(self.jet_image[y,x].copy())

        graphs.append(graph)
      except Exception as e:
        facereclib.utils.error("Could not extract feature for patch %i: %s" %(i, e))
    sys.stdout.write("\n")

    if self.number_of_means is None:
      # just store the jets from ALL training patches
      self.graphs = numpy.array(graphs)
    else:
      # cluster the jets for each node
      position_count = self.number_of_predictions()/2
      self.graphs = numpy.zeros((self.number_of_means, position_count, 2, self.gwt.number_of_kernels))
      for i in range(position_count):
        facereclib.utils.info("Clustering jets for index %d" % i)
        jets = [graph[i] for graph in graphs]
        means = self.cluster(jets)
        for n in range(self.number_of_means):
          self.graphs[n,i] = means[n]



  def save(self, hdf5):
    LocalModel.save(self, hdf5)
    hdf5.set("Graphs", self.graphs)
    hdf5.create_group("Gabors")
    hdf5.cd("Gabors")
    self.gwt.save(hdf5)
    hdf5.cd("..")


  def load(self, hdf5):
    LocalModel.load(self, hdf5)
    self.graphs = hdf5.read("Graphs")
    hdf5.cd("Gabors")
    self.gwt = bob.ip.GaborWaveletTransform(hdf5)
    hdf5.cd("..")
    self.jet_image = self.gwt.empty_jet_image(numpy.ndarray([self.patch_size[i] + 2 * self.offset[i] for i in (0,1)]), True)


  def init_prediction(self, image):
    # perform gwt on image patch
    self.gwt.compute_jets(image, self.jet_image, True)

  def compute_shift(self, prediction):
    # computes the best shift for the current prediction
    shift = []
    # iterate over all positions
    for a in range(0, len(prediction), 2):
      # get the average position as a starting point
      y = prediction[a]
      x = prediction[a+1]

#      if a == 16: import ipdb; ipdb.set_trace()
      # compute similarities and disparities for all stored graphs
      similarities = []
      disparities = []
      for graph in self.graphs:
        reference_jet = graph[a/2]
        y = max(min(y,self.jet_image.shape[0]-1), 0)
        x = max(min(x,self.jet_image.shape[1]-1), 0)
        target_jet = self.jet_image[y,x]
        similarities.append(self.similarity(reference_jet, target_jet))
        disparities.append(self.similarity.disparity())

#      if a == 16: import ipdb; ipdb.set_trace()
      # get a sorted list of similarities and corresponding disparities
      best_similarity_indices = numpy.argsort(similarities)[:100]
#      similarities = [similarities[i] for i in best_similarity_indices]
      disparities = [disparities[i] for i in reversed(best_similarity_indices)]

      # histogram the directions of the disparities
      histogram = [[],[],[],[]]
      for disp in disparities:
        i = (2 if disp[0] < 0 else 0) + (1 if disp[1] < 0 else 0)
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

      shift.append(disparity[0])
      shift.append(disparity[1])

    return shift


