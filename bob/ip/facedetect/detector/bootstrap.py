import numpy
import facereclib
from .io import load, save
from .sampler import Sampler
from .. import FeatureExtractor
import os

import bob.ip.base
import bob.learn.boosting

class Bootstrap:
  """This class deals with selecting new training examples for each boosting round."""
  def __init__(self, number_of_rounds = 7, number_of_weak_learners_in_first_round = 8, number_of_positive_examples_per_round = 5000, number_of_negative_examples_per_round = 5000, init_with_average = False):
    self.m_number_of_rounds = number_of_rounds
    self.m_number_of_weak_learners_per_round = [number_of_weak_learners_in_first_round * 2**i for i in range(number_of_rounds)]
    self.m_number_of_positive_examples_per_round = number_of_positive_examples_per_round
    self.m_number_of_negative_examples_per_round = number_of_negative_examples_per_round
    self.m_init_with_average = init_with_average

  def __call__(self, trainer, sampler, feature_extractor, model = None, filename="bootstrapped_model.hdf5"):
    training_data = numpy.ndarray((0,feature_extractor.number_of_features), numpy.uint16)
    training_labels = None
    mean = None
    variance = None
    for b in range(self.m_number_of_rounds):
      # get new data
      facereclib.utils.info("Getting new data for bootstrapping round %d" % (b+1))
      new_data, new_labels, new_means, new_variances = sampler.get(feature_extractor, model, self.m_number_of_positive_examples_per_round, self.m_number_of_negative_examples_per_round, delete_samples=True, compute_means_and_variances=True, loss_function=trainer.get_loss_function())

      training_data = numpy.append(training_data, new_data, axis=0)
      if training_labels is None:
        training_labels = new_labels
      else:
        training_labels = numpy.append(training_labels, new_labels, axis=0)

      temp_file = "%s_round_%d.hdf5" % (os.path.splitext(filename)[0], b+1)
      if os.path.exists(temp_file):
        facereclib.utils.info("Loading already computed stage from %s." % temp_file)
        model, feature_extractor, mean, variance = load(temp_file)

      else:
        if self.m_init_with_average and model is None and b == 0:
          facereclib.utils.info("Initializing first weak machine with the average of the training labels.")
          # compute average of labels and initialize the first weak machine with it
          model = bob.learn.boosting.BoostedMachine()
          average = numpy.mean(new_labels, axis=0)
          lut = numpy.ones((feature_extractor.number_of_labels, average.shape[0]))
          lut[:,average>0] = -1
          average = -numpy.abs(average)
          weak_machine = bob.learn.boosting.LUTMachine(lut, numpy.zeros(average.shape[0], numpy.int32))
          model.add_weak_machine(weak_machine, average)
          save("%s_round_%d.hdf5" % (os.path.splitext(filename)[0], 0), model, feature_extractor, mean, variance)


        if mean is None:
          mean = (min(new_means), max(new_means))
          variance = (min(new_variances), max(new_variances))
        elif new_means.size:
          mean = (min(min(new_means), mean[0]), max(max(new_means), mean[1]))
          variance = (min(min(new_variances), variance[0]), max(max(new_variances), variance[1]))

        facereclib.utils.info("Starting training with %d examples" % (training_data.shape[0]))
        model = trainer.train(training_data, training_labels, self.m_number_of_weak_learners_per_round[b], model)

        # write model and extractor to temporary file to be able to catch up later
        save(temp_file, model, feature_extractor, mean, variance)

      feature_extractor.model_indices = model.indices

    # finally, return the trained model
    return model, mean, variance


  def coarse_to_fine_feature_selection(self, trainer, sampler, feature_extractor, filename = "ctf_model.hdf5", number_of_rounds = 3):
    def h(lbp):
      # LBP hash function
      return "%02d-%02d" % (lbp.block_size if lbp.is_multi_block_lbp else lbp.radii)

    patch_scale_factor = 2
    scale = pow(patch_scale_factor, - number_of_rounds + 1)
    scaled_sampler = sampler.scale_self(scale)
    for r in range(number_of_rounds):

      temp_filename = "%s_scale-%1.2f.hdf5"% (os.path.splitext(filename)[0], scale)
      last_temp_filename = "%s_round_%d.hdf5" % (os.path.splitext(temp_filename)[0], self.m_number_of_rounds)
      if os.path.exists(last_temp_filename):
        facereclib.utils.info("Loading result file %s of scale %1.2f" % (last_temp_filename, scale))
        model, feature_extractor, mean, variance = load(last_temp_filename)
      else:
        facereclib.utils.info("Starting CTFFS with patch size %s" % str(feature_extractor.patch_size))
        model, mean, variance = self.__call__(trainer, scaled_sampler, feature_extractor, None, os.path.splitext(filename)[0] + "_scale-%1.2f"%scale + os.path.splitext(filename)[1])

      assert feature_extractor.patch_size == scaled_sampler.m_patch_size

      scale *= patch_scale_factor
      if r == number_of_rounds - 1:
        # we are done.
        return model, feature_extractor
      elif r == number_of_rounds - 2:
        # we can use the default sampler
        assert abs(1. - scale) < 1e-8
        scaled_sampler = sampler
      else:
        # we have to scale the sampler
        scaled_sampler = sampler.scale_self(scale)

      # create a feature extractor that is composed of the features used by the model
      if self.m_init_with_average:
        model_indices = sorted(set(model.feature_indices(1)))
      else:
        model_indices = sorted(set(model.indices))

      lbps = {}
      positions = {}
      for index in model_indices:
        try:
          lbp = feature_extractor.extractor(index)
        except Exception as e:
          print "model index", index, "seems to be unaccessible:", e
        position = feature_extractor.offset(index)
        # if the old position is odd, add 1 since the center is shifted
        new_position = (position[0] * patch_scale_factor + position[0] % 2, position[1] * patch_scale_factor + position[1] % 2)

        # scale up the lbp and the positions, and add features with slightly modified extends
        for y in (-1,0,1):
          for x in (-1,0,1):
            new_lbp = bob.ip.base.LBP(lbp)
            new_shape = (patch_scale_factor * lbp.block_size[0] + y, patch_scale_factor * lbp.block_size[1] + x)
            if new_lbp.is_multi_block_lbp:
              if lbp.block_overlap[0] != 0:
                new_lbp.set_block_size_and_overlap(new_shape, (new_shape[0]-1, new_shape[1]-1))
              else:
                new_lbp.block_size = new_shape
            else:
              new_lbp.radii = new_shape

            # check if this LBP is valid for our patch size
            lbp_shape = new_lbp.get_lbp_shape(scaled_sampler.m_patch_size, False)
            lbp_offset = new_lbp.offset
            if lbp_shape[0] == 0 or lbp_shape[1] == 0:
              facereclib.utils.debug("Cannot use LBP extractor with shape %s since it is outside the possible range" % str(new_shape))
            elif new_position[0] < lbp_offset[0] or new_position[1] < lbp_offset[1] or new_position[0] >= lbp_shape[0]+lbp_offset[0] or new_position[1] >= lbp_shape[1]+lbp_offset[1]:
              facereclib.utils.debug("Cannot use LBP extractor with shape %s and offset %s since it is outside the possible range" % (str(new_shape), str(new_position)))
            else:
              if h(new_lbp) in positions and new_position not in positions[h(new_lbp)]:
                positions[h(new_lbp)].append(new_position)
              else:
                positions[h(new_lbp)] = [new_position]
                lbps[h(new_lbp)] = new_lbp

      # overwrite the feature extractor
      feature_extractor = FeatureExtractor(scaled_sampler.m_patch_size)
      for k in sorted(lbps.keys()):
        feature_extractor.append(lbps[k], positions[k])

      facereclib.utils.info("Generated new feature extractor with %d features" % feature_extractor.number_of_features)

#      trainer.m_trainer = bob.learn.boosting.trainer.LUTTrainer(feature_extractor.number_of_labels, feature_extractor.number_of_features, trainer.m_trainer.m_number_of_outputs, trainer.m_trainer.m_selection_type)
      trainer.m_trainer = bob.learn.boosting.trainer.LUTTrainer(feature_extractor.number_of_labels, trainer.m_trainer.number_of_outputs, trainer.m_trainer.selection_type)

