import bob
import numpy
import facereclib
import sys
import math

from .LocalModel import LocalModel

def adjust_phase(phase):
  return phase - (2.*math.pi)*round(phase / (2.*math.pi))

class JetStatistics (LocalModel):

  def __init__(self, patch_size = (96,80), gwt = bob.ip.GaborWaveletTransform(7), subspace_size = 0.98, offset = (30,30)):
    LocalModel.__init__(self, patch_size, subspace_size, offset)

    self.gwt = gwt
    self.similarity = bob.machine.GaborJetSimilarity(bob.machine.gabor_jet_similarity_type.DISPARITY, gwt)
    self.jet_image = self.gwt.empty_jet_image(numpy.ndarray([patch_size[i] + 2 * offset[i] for i in (0,1)]), True)


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

    # compute average jet from training jets
    landmark_count = train_annotations[0].shape[0]/2
    averager = bob.machine.GaborGraphMachine()
    jet_length = self.gwt.number_of_kernels
    self.models = []
    for l in range(landmark_count):
      facereclib.utils.info("Computing statistics for landmark %d" % l)
      # collect jets from all training images and put them into the desired structure
      jets = [graph[l] for graph in graphs]
      # average the jets
      average = numpy.ndarray([1, 2, jet_length])
      averager.average(numpy.expand_dims(numpy.array(jets), axis=1), average)
      mean_jet = average[0]
      # compute phase shift towards average jet
#      phase_shifted_jets = [self.similarity.shift_phase(jet, mean_jet) for jet in jets]
#      # average these phase corrected jets again
#      averager.average(numpy.expand_dims(numpy.array(phase_shifted_jets), axis=1), average)
#      mean_jet = average[0]

      # now, compute statistics of abs and phase values
      mean_abs = mean_jet[0,:]
      variance_abs = [sum((jet[0][j] - mean_abs[j])**2 for jet in jets) / (len(jets)-1) for j in range(jet_length)]

      mean_phase = mean_jet[1,:]
      # variant A: we compute the statistics of the already phase shifted jets (requires to perform phase shifting during evaluation as well)
#      variance_phase = [sum(adjust_phase(jet[1][j] - mean_phase[j])**2 for jet in phase_shifted_jets) / (len(jets)-1) for j in range(jet_length)]

      # variant B: we compute the statistics of the already original jets
      variance_phase = [sum(adjust_phase(jet[1][j] - mean_phase[j])**2 for jet in jets) / (len(jets)-1) for j in range(jet_length)]

      # store the information
      self.models.append((mean_abs, variance_abs, mean_phase, variance_phase))


  def save(self, hdf5):
    LocalModel.save(self, hdf5)
    for i in range(len(self.models)):
      hdf5.create_group("Landmark%d"%i)
      hdf5.cd("Landmark%d"%i)
      hdf5.set("MeanAbs", self.models[i][0])
      hdf5.set("VarAbs", self.models[i][1])
      hdf5.set("MeanPhase", self.models[i][2])
      hdf5.set("VarPhase", self.models[i][3])
      hdf5.cd("..")
    hdf5.create_group("Gabors")
    hdf5.cd("Gabors")
    self.gwt.save(hdf5)
    hdf5.cd("..")


  def load(self, hdf5):
    LocalModel.load(self, hdf5)
    self.models = []
    a = 0
    while hdf5.has_group("Landmark%i"%a):
      hdf5.cd("Landmark%i"%a)
      self.models.append((hdf5.read("MeanAbs"), hdf5.read("VarAbs"), hdf5.read("MeanPhase"), hdf5.read("VarPhase")))
      hdf5.cd("..")
      a += 1
    hdf5.cd("Gabors")
    self.gwt = bob.ip.GaborWaveletTransform(hdf5)
    hdf5.cd("..")
    self.jet_image = self.gwt.empty_jet_image(numpy.ndarray([self.patch_size[i] + 2 * self.offset[i] for i in (0,1)]), True)


  def init_prediction(self, image):
    # perform gwt on image patch
    self.gwt.compute_jets(image, self.jet_image, True)

  def disparity(self, landmark, jet):
    # estimates the disparity for the given jet
    mean_abs, variance_abs, mean_phase, variance_phase = self.models[landmark]
    jet_len = jet.shape[1]

    # shift phases towards mean
#    phase_shifted_jet = self.similarity.shift_phase(jet, numpy.array([mean_abs, mean_phase]))

    # compute confidences and phase differences
#    confidences = [phase_shifted_jet[0,j] * mean_abs[j] for j in range(jet_len)]
#    phase_differences = [mean_phase[j] - phase_shifted_jet[1,j] for j in range(jet_len)]

    # compute confidences and phase differences
    confidences = [jet[0,j] * mean_abs[j] for j in range(jet_len)]
    phase_differences = [mean_phase[j] - jet[1,j] for j in range(jet_len)]

    # compute disparity
    gamma_y_y, gamma_y_x, gamma_x_x, phi_y, phi_x = 0., 0., 0., 0., 0.
    disparity = [0., 0.]
    kernels = [self.gwt.kernel_frequency(j) for j in range(jet_len)]

    j = jet_len
    for scale in reversed(range(self.gwt.number_of_scales)):
      for direction in reversed(range(self.gwt.number_of_directions)):
        j = j - 1
        kjy, kjx = kernels[j]
        conf = confidences[j]
        diff = phase_differences[j]
        var = variance_phase[j]

        # totalize gamma matrix
        gamma_y_y += conf * kjy * kjy / var
        gamma_y_x += conf * kjy * kjx / var
        gamma_x_x += conf * kjx * kjx / var

        # totalize phi vector
        # estimate the number of cycles that we are off (using the current estimation of the disparity
        n = round((diff - disparity[0] * kjy - disparity[1] * kjx) / (2.*math.pi));
        # totalize corrected phi vector elements
        phi_y += conf * (diff - n * 2. * math.pi) * kjy / var
        phi_x += conf * (diff - n * 2. * math.pi) * kjx / var

      # re-calculate disparity as d=\Gamma^{-1}\Phi of the (low frequency) wavelet scales that we used up to now
      gamma_det = gamma_x_x * gamma_y_y - gamma_y_x**2
      disparity[0] = (gamma_x_x * phi_y - gamma_y_x * phi_x) / gamma_det
      disparity[1] = (gamma_y_y * phi_x - gamma_y_x * phi_y) / gamma_det

    return disparity


  def compute_shift(self, prediction):
    # computes the best shift for the current prediction
    shift = []
    # iterate over all landmarks
    for a in range(0, len(prediction), 2):
      # get the current position as a starting point
      y = prediction[a]
      x = prediction[a+1]

      # compute the disparity for the given jet
      disparity = self.disparity(a/2, self.jet_image[y,x])

      # this disparity is the shift
      shift.append(disparity[0])
      shift.append(disparity[1])

    return shift


