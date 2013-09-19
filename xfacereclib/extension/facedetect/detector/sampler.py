
import math
import numpy
import bob
import facereclib
import itertools

from ..utils import BoundingBox


class Sampler:
  """This class generates and contains bounding boxes positive and negative examples used for face detection."""

  def __init__(self, patch_size = (24,20), scale_factor = math.pow(2., -1./4.), first_scale = 0.5, distance = 2, similarity_thresholds = (0.3, 0.7)):
    """Generates an example extractor for the given patch size.

    Parameters:

    patch_size:
      the size of the path to extract

    scale_factor:
      image pyramids are computed using the given scale factor between two scales.
      warning: The original code used a MUCH HIGHER density of scales (where the scale factor was computed automatically).

    distance:
      the distance in both horizontal and vertical direction to generate samples

    similarity_thresholds:
      two patches will be compared:the (scaled) annotated patch and the (shifted) extracted patch
      if the similarity is lower than the first value of the similarity_thresholds tuple, it will be accepted as negative example
      if the similarity is higher than the second value of the similarity_thresholds tuple, it will be accepted as positive example
      otherwise the patch will be rejected.

    """

    self.m_scales = []
    self.m_scaled_images = []
    self.m_positives = []
    self.m_negatives = []

    self.m_patch_box = BoundingBox("direct", topleft=(0,0), bottomright=patch_size)
    self.m_scale_factor = scale_factor
    self.m_first_scale = first_scale
    self.m_distance = distance
    self.m_similarity_thresholds = similarity_thresholds


  def add(self, image, ground_truth, number_of_positives_per_scale = None, number_of_negatives_per_scale = None):
    """Adds positive and negative examples from the given image, using the given ground_truth bounding boxes."""
    from ..utils import irnd

    current_scale_power = 0.
    while True:
      # scale the image
      scale = self.m_first_scale * math.pow(self.m_scale_factor, current_scale_power)
      current_scale_power += 1.

      scaled_image = bob.ip.scale(image, scale)
      if scaled_image.shape[0] < self.m_patch_box.m_bottom or scaled_image.shape[1] < self.m_patch_box.m_top:
        # image is too small since there is not enough space to put in the complete patch box
        break

      # adapt distance so that scaled images are scanned tighter
      distance = int(math.ceil(self.m_distance*scale / self.m_first_scale))
#      facereclib.utils.debug("Scaled image size %s" %str(scaled_image.shape))
      scaled_gt = [gt.scale(scale) for gt in ground_truth]
      positives = []
      negatives = []

      # iterate over all possible positions in the image
      for y in range(0, scaled_image.shape[0]-self.m_patch_box.m_bottom, distance):
        for x in range(0, scaled_image.shape[1]-self.m_patch_box.m_right, distance):
          # create bounding box for the image
          bb = self.m_patch_box.shift(y,x)

          # check if the patch is a positive example
          positive = False
          negative = True
          for gt in scaled_gt:
            similarity = bb.similarity(gt)
            if similarity > self.m_similarity_thresholds[1]:
#              facereclib.utils.debug("Found positive bounding box %s with similarity value %f" % (str(bb), similarity))
              positive = True
              break

            if similarity > self.m_similarity_thresholds[0]:
#              facereclib.utils.debug("Rejecting negative bounding box %s  -- '%s' with similarity value %f" % (str(bb), str(gt), similarity))
              negative = False
              break

          if positive:
            positives.append(bb)
          elif negative:
            negatives.append(bb)
          # else: ignore patch

      # at the end, add found patches
      self.m_scales.append(scale)
      self.m_scaled_images.append(scaled_image)
      self.m_positives.append([positives[i] for i in facereclib.utils.quasi_random_indices(len(positives), number_of_positives_per_scale)])
      self.m_negatives.append([negatives[i] for i in facereclib.utils.quasi_random_indices(len(negatives), number_of_negatives_per_scale)])


  def get(self, feature_extractor, model = None, maximum_number_of_positives = None, maximum_number_of_negatives = None, delete_samples = False):
    """Returns a pair of features and labels that can be used for training, after extracting the features using the given feature extractor.
    If number_of_positives and/or number_of_negatives are given, the number of examples is limited to these numbers."""
    # get the maximum number of examples
    all_pos = sum(len(pos) for pos in self.m_positives)
    all_neg = sum(len(neg) for neg in self.m_negatives)
    num_pos = all_pos if maximum_number_of_positives is None else min(maximum_number_of_positives, all_pos)
    num_neg = all_neg if maximum_number_of_negatives is None else min(maximum_number_of_negatives, all_neg)

    facereclib.utils.info("Extracting %d (%d) positive and %d (%d) negative examples" % (num_pos, all_pos, num_neg, all_neg))

    # create feature and labels as required for training
    features = numpy.ndarray((num_pos + num_neg, feature_extractor.number_of_features()), numpy.uint16)
    labels = numpy.ones((num_pos + num_neg, ), numpy.float64)

    # get the positive and negative examples
    if model is None:
      # collect positive and negative examples
      all_positive_examples = [(i,j) for i, positives in enumerate(self.m_positives) for j in range(len(positives))]
      all_negative_examples = [(i,j) for i, negatives in enumerate(self.m_negatives) for j in range(len(negatives))]

      # simply compute a random subset of both lists
      # (for testing purposes, this is quasi-random)
      used_positive_examples = [all_positive_examples[i] for i in facereclib.utils.quasi_random_indices(len(all_positive_examples), num_pos)]
      used_negative_examples = [all_negative_examples[i] for i in facereclib.utils.quasi_random_indices(len(all_negative_examples), num_neg)]
    else:
      # compute the prediction error of the current classifier for all remaining
      positive_values = []
      negative_values = []
      feature_vector = numpy.zeros((feature_extractor.number_of_features(),), numpy.uint16)
      for image_index, image in enumerate(self.m_scaled_images):
        # prepare for current scaled image
        feature_extractor.prepare(self.m_scaled_images[image_index])
        for bb_index, bb in enumerate(self.m_positives[image_index]):
          # extract the features for the current bounding box
          feature_extractor.extract_single(bb, feature_vector)
          # compute the current prediction of the model
          positive_values.append((model(feature_vector)[0], image_index, bb_index))
        for bb_index, bb in enumerate(self.m_negatives[image_index]):
          # extract the features for the current bounding box
          feature_extractor.extract_single(bb, feature_vector)
          # compute the current prediction of the model
          negative_values.append((model(feature_vector)[0], image_index, bb_index))

      # get the prediction errors (lowest for pos. class and highest for neg. class)
      positive_values = sorted(positive_values)[:num_pos]
      negative_values = sorted(negative_values, reverse=True)[:num_neg]
      used_positive_examples = [pos[1:] for pos in sorted(positive_values)]
      used_negative_examples = [neg[1:] for neg in sorted(negative_values)]


    last_image_index = -1
    i = 0
    for image_index, patch_index in used_positive_examples:
      # prepare for this image, if it has changed
      if image_index != last_image_index:
        last_image_index = image_index
        feature_extractor.prepare(self.m_scaled_images[image_index])
      # extract and append features
      feature_extractor.extract(self.m_positives[image_index][patch_index], features, i)
      i += 1

    # append positive examples
    for image_index, patch_index in used_negative_examples:
      # prepare for this image, if it has changed
      if image_index != last_image_index:
        last_image_index = image_index
        feature_extractor.prepare(self.m_scaled_images[image_index])
      # extract and append features
      feature_extractor.extract(self.m_negatives[image_index][patch_index], features, i)
      labels[i] = -1.
      i += 1

    # finally, delete all examples that we returned
    if delete_samples:
      # TODO: implement faster versions of this; currently it is O(n^2)
      for image_index, patch_index in sorted(used_positive_examples, reverse=True):
        del self.m_positives[image_index][patch_index]
      for image_index, patch_index in sorted(used_negative_examples, reverse=True):
        if len(self.m_negatives[image_index]) <= patch_index:
          print image_index, len(self.m_negatives[image_index]), patch_index
        del self.m_negatives[image_index][patch_index]

    # return the collected features and labels
    return (features, labels)


  def iterate(self, image, feature_extractor):
    """Scales the given image and extracts bounding boxes, computes the features for the given feature extractor and returns an ITERATOR returning a pair of bounding_box and feature.
    """

    # for now, we just call the add function and return the latest results
    assert not self.m_scales
    assert not self.m_scaled_images
    assert not self.m_positives
    assert not self.m_negatives

    self.add(image, [])

    feature_vector = numpy.zeros(feature_extractor.number_of_features(), numpy.uint16)
    for scaled_image, boxes, scale in itertools.izip(self.m_scaled_images, self.m_negatives, self.m_scales):
      # prepare the feature extractor to extract features from the given image
      bob.io.save(scaled_image.astype(numpy.uint8), "/scratch/mguenther/temp/scaled/im_%f.png"%scale)
      feature_extractor.prepare(scaled_image)
      # iterate over all boxes
      for box in boxes:
        # extract features for
        feature_extractor.extract_single(box, feature_vector)
        yield box.scale(1./scale), feature_vector

    # at the end, clean up the mess
    self.m_scales = []
    self.m_scaled_images = []
    self.m_positives = []
    self.m_negatives = []


  def _write(self, name="/scratch/mguenther/temp/examples/image_%i.png", write_positives = True, write_negatives = False):
    """Writes the positive (and negative) training examples to the file with the given file name."""
    if write_positives:
      i = 0
      for index, image in enumerate(self.m_scaled_images):
        for bb in self.m_positives[index]:
           bob.io.save(image[bb.m_top:bb.m_bottom, bb.m_left:bb.m_right].astype(numpy.uint8), name % i)
           i += 1

