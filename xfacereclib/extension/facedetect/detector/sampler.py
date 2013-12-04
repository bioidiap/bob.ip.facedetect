
import math
import numpy
import bob
import facereclib
import itertools

from ..utils import BoundingBox
from .._features import BoundingBox as CppBoundingBox


import threading


class Sampler:
  """This class generates and contains bounding boxes positive and negative examples used for face detection."""

  def __init__(self, patch_size = (24,20), scale_factor = math.pow(2., -1./4.), first_scale = 0.5, distance = 2, similarity_thresholds = (0.3, 0.7), mirror_samples=False, cpp_implementation=True, number_of_parallel_threads=1):
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
    self.m_images = []
    self.m_positives = []
    self.m_negatives = []

    if cpp_implementation:
      self.m_patch_box = CppBoundingBox(0, 0, patch_size[0], patch_size[1])
    else:
      self.m_patch_box = BoundingBox("direct", topleft=(0,0), bottomright=(patch_size[0]-1, patch_size[1]-1))
    self.m_scale_factor = scale_factor
    self.m_first_scale = first_scale
    self.m_distance = distance
    self.m_similarity_thresholds = similarity_thresholds
    self.m_number_of_parallel_threads = number_of_parallel_threads
    self.m_mirror_samples = mirror_samples


  def _scales(self, image):
    current_scale_power = 0.
    while True:
      # scale the image
      scale = self.m_first_scale * math.pow(self.m_scale_factor, current_scale_power)
      current_scale_power += 1.
      scaled_image_shape = bob.ip.get_scaled_output_shape(image, scale)
      if scaled_image_shape[0] <= self.m_patch_box.bottom or scaled_image_shape[1] <= self.m_patch_box.right:
        # image is too small since there is not enough space to put in the complete patch box
        break

      yield scale, scaled_image_shape


  def _sample(self, scaled_image_shape):
    """Returns an iterator that iterates over the sampled positions in the image."""
    for y in range(0, scaled_image_shape[0]-self.m_patch_box.bottom-1, self.m_distance):
      for x in range(0, scaled_image_shape[1]-self.m_patch_box.right-1, self.m_distance):
        # create bounding box for the image
        yield self.m_patch_box.shift(y,x)



  def add(self, image, ground_truth, number_of_positives_per_scale = None, number_of_negatives_per_scale = None):
    """Adds positive and negative examples from the given image, using the given ground_truth bounding boxes."""
    from ..utils import irnd

    # remeber the image
    self.m_images.append(image)
    # remember the possible scales for this image
    self.m_scales.append([])
    # remember, which patches of which image of which scale is a positive or a negative example
    self.m_positives.append([])
    self.m_negatives.append([])

    for scale, scaled_image_shape in self._scales(image):

#      facereclib.utils.debug("Scaled image size %s" %str(scaled_image.shape))
      scaled_gt = [gt.scale(scale) for gt in ground_truth]
      positives = []
      negatives = []

      # iterate over all possible positions in the image
      for bb in self._sample(scaled_image_shape):
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
      self.m_scales[-1].append(scale)
      self.m_positives[-1].append([positives[i] for i in facereclib.utils.quasi_random_indices(len(positives), number_of_positives_per_scale)])
      self.m_negatives[-1].append([negatives[i] for i in facereclib.utils.quasi_random_indices(len(negatives), number_of_negatives_per_scale)])


  def get(self, feature_extractor, model = None, maximum_number_of_positives = None, maximum_number_of_negatives = None, delete_samples = False, compute_means_and_variances = False):

    def _get_parallel(pos, neg, first, last):
      """Extracts the feature for the given set of feature type and return the model response"""
      # get a copy of the feature extractors to assure thread-safety
      fex = feature_extractor.__class__(feature_extractor)
      feature_vector = numpy.zeros((fex.number_of_features,), numpy.uint16)
      for image_index in range(first, last):
        for scale_index, scale in enumerate(self.m_scales[image_index]):
          # prepare for current image in current scale
          fex.prepare(self.m_images[image_index], scale)
          # extract features
          for bb_index, bb in enumerate(self.m_positives[image_index][scale_index]):
            # extract the features for the current bounding box
            fex.extract_single_p(bb, feature_vector)
            # compute the current prediction of the model
            pos.append((model.forward_p(feature_vector), image_index, scale_index, bb_index))
          for bb_index, bb in enumerate(self.m_negatives[image_index][scale_index]):
            # extract the features for the current bounding box
            fex.extract_single_p(bb, feature_vector)
            # compute the current prediction of the model
            neg.append((model.forward_p(feature_vector), image_index, scale_index, bb_index))


    def _extract_parallel(examples, bounding_boxes, dataset, first, last, offset, compute_means = False, means=[], variances=[], mirror_offset = 0):
      last_image_index = -1
      last_scale_index = 1
      fex = feature_extractor.__class__(feature_extractor)
      if mirror_offset:
        mex = feature_extractor.__class__(feature_extractor)
      for index in range(first, last):
        image_index, scale_index, bb_index = examples[index]
        # prepare for this image, if it has changed
        if last_scale_index != scale_index or last_image_index != image_index:
          last_scale_index = scale_index
          last_image_index = image_index
          fex.prepare_p(self.m_images[image_index], self.m_scales[image_index][scale_index], compute_means_and_variances)
          if mirror_offset:
            mex.prepare(self.m_images[image_index][:,::-1].copy(), self.m_scales[image_index][scale_index])
        # extract and append features
        bb = bounding_boxes[image_index][scale_index][bb_index]
        fex.extract_p(bb, dataset, index + offset)
        if mirror_offset:
          mex.extract_p(bb.mirror_x(mex.image.shape[1]), dataset, index + offset + mirror_offset)
        if compute_means:
          m,v = fex.mean_and_variance(bb)
          means[index+offset] = m
          variances[index+offset] = v

    def _get_all(pos_or_neg):
      for image_count in range(len(pos_or_neg)):
        for scale_count in range(len(pos_or_neg[image_count])):
          for bb_count in range(len(pos_or_neg[image_count][scale_count])):
            yield (image_count, scale_count, bb_count)

    """Returns a pair of features and labels that can be used for training, after extracting the features using the given feature extractor.
    If number_of_positives and/or number_of_negatives are given, the number of examples is limited to these numbers."""
    # get the maximum number of examples
    pos_count = len(list(_get_all(self.m_positives)))
    neg_count = len(list(_get_all(self.m_negatives)))
    if self.m_mirror_samples:
      pos_count *= 2
      neg_count *= 2
    num_pos = pos_count if maximum_number_of_positives is None else min(maximum_number_of_positives, pos_count)
    num_neg = neg_count if maximum_number_of_negatives is None else min(maximum_number_of_negatives, neg_count)


    # create feature and labels as required for training
    dataset = numpy.ndarray((num_pos + num_neg, feature_extractor.number_of_features), numpy.uint16)
    labels = numpy.ones((num_pos + num_neg, ), numpy.float64)
    means = numpy.ndarray((num_pos/2 if self.m_mirror_samples else num_pos, ), numpy.float64)
    variances = numpy.ndarray((num_pos/2 if self.m_mirror_samples else num_pos, ), numpy.float64)

    # get the positive and negative examples
    if model is None:
      # collect positive and negative examples
      all_positive_examples = list(_get_all(self.m_positives))
      all_negative_examples = list(_get_all(self.m_negatives))

      # simply compute a random subset of both lists
      # (for testing purposes, this is quasi-random)
      used_positive_examples = [all_positive_examples[i] for i in facereclib.utils.quasi_random_indices(len(all_positive_examples), num_pos/2 if self.m_mirror_samples else num_pos)]
      used_negative_examples = [all_negative_examples[i] for i in facereclib.utils.quasi_random_indices(len(all_negative_examples), num_neg/2 if self.m_mirror_samples else num_neg)]
    else:

      facereclib.utils.info("Computing classification results for %d positive and %d negative training items" % (pos_count, neg_count))
      # compute the prediction error of the current classifier for all remaining
      if self.m_number_of_parallel_threads == 1:

        positive_values = []
        negative_values = []
        feature_vector = numpy.zeros((feature_extractor.number_of_features,), numpy.uint16)
        for image_index, image in enumerate(self.m_images):
          for scale_index, scale in enumerate(self.m_scales[image_index]):
            # prepare for current scaled image
            feature_extractor.prepare(image, scale)
            for bb_index, bb in enumerate(self.m_positives[image_index][scale_index]):
              # extract the features for the current bounding box
              feature_extractor.extract_single(bb, feature_vector)
              # compute the current prediction of the model
              positive_values.append((model(feature_vector), image_index, scale_index, bb_index))
            for bb_index, bb in enumerate(self.m_negatives[image_index][scale_index]):
              # extract the features for the current bounding box
              feature_extractor.extract_single(bb, feature_vector)
              # compute the current prediction of the model
              negative_values.append((model(feature_vector), image_index, scale_index, bb_index))

      else:

        # parallel implementation
        number_of_indices = len(self.m_images)
        indices = [i * number_of_indices / self.m_number_of_parallel_threads for i in range(self.m_number_of_parallel_threads)] + [number_of_indices]

        parallel_positive_results = [[] for i in range(self.m_number_of_parallel_threads)]
        parallel_negative_results = [[] for i in range(self.m_number_of_parallel_threads)]
        parallel_means = [[] for i in range(self.m_number_of_parallel_threads)]

        threads = [threading.Thread(target=_get_parallel, args=(parallel_positive_results[i], parallel_negative_results[i], indices[i], indices[i+1])) for i in range(self.m_number_of_parallel_threads)]
        [t.start() for t in threads]
        [t.join() for t in threads]
        positive_values = [x for pos in parallel_positive_results for x in pos]
        negative_values = [x for neg in parallel_negative_results for x in neg]


      # get the prediction errors (lowest for pos. class and highest for neg. class)
      positive_values = sorted(positive_values)[:num_pos/2 if self.m_mirror_samples else num_pos]
      negative_values = sorted(negative_values, reverse=True)[:num_neg/2 if self.m_mirror_samples else num_neg]
      used_positive_examples = [pos[1:] for pos in positive_values]
      used_negative_examples = [neg[1:] for neg in negative_values]


    facereclib.utils.info("Extracting %d (%d) positive and %d (%d) negative examples" % (num_pos, pos_count, num_neg, neg_count))

    # We have decided, which patches to take,
    # Now, extract the features...
    if self.m_number_of_parallel_threads == 1:
      last_image_index = -1
      last_scale_index = -1
      i = 0
      if self.m_mirror_samples:
        mirror_extractor = feature_extractor.__class__(feature_extractor)
        mirror_offset = len(used_positive_examples)
      # append positive examples
      for image_index, scale_index, bb_index in used_positive_examples:
        # prepare for this image, if it has changed
        if last_scale_index != scale_index or last_image_index != image_index:
          last_scale_index = scale_index
          last_image_index = image_index
          feature_extractor.prepare(self.m_images[image_index], self.m_scales[image_index][scale_index], compute_means_and_variances)
          if self.m_mirror_samples:
            # prepare for the mirrored image
            mirror_extractor.prepare(self.m_images[image_index][:,::-1].copy(), self.m_scales[image_index][scale_index])
        # extract and append features
        bb = self.m_positives[image_index][scale_index][bb_index]
        feature_extractor.extract(bb, dataset, i)
        if self.m_mirror_samples:
          mirror_extractor.extract(bb.mirror_x(mirror_extractor.image.shape[1]), dataset, i + mirror_offset)
        if compute_means_and_variances:
          m,v = feature_extractor.mean_and_variance(bb)
          means[i] = m
          variances[i] = v
        i += 1

      if self.m_mirror_samples:
        mirror_extractor = feature_extractor.__class__(feature_extractor)
        i += mirror_offset
        mirror_offset = len(used_negative_examples)
      # append negative examples
      for image_index, scale_index, bb_index in used_negative_examples:
        # prepare for this image, if it has changed
        if last_scale_index != scale_index or last_image_index != image_index:
          last_scale_index = scale_index
          last_image_index = image_index
          feature_extractor.prepare(self.m_images[image_index], self.m_scales[image_index][scale_index])
          if self.m_mirror_samples:
            # prepare for the mirrored image
            mirror_extractor.prepare(self.m_images[image_index][:,::-1].copy(), self.m_scales[image_index][scale_index])
        # extract and append features
        bb = self.m_negatives[image_index][scale_index][bb_index]
        feature_extractor.extract(bb, dataset, i)
        labels[i] = -1.
        if self.m_mirror_samples:
          mirror_extractor.extract(bb.mirror_x(mirror_extractor.image.shape[1]), dataset, i + mirror_offset)
          labels[i + mirror_offset] = -1.
        i += 1
    else: # parallel implementation
      # positives
      pos_mirror_offset = len(used_positive_examples) if self.m_mirror_samples else 0
      number_of_indices = len(used_positive_examples)
      indices = [i * number_of_indices / self.m_number_of_parallel_threads for i in range(self.m_number_of_parallel_threads)] + [number_of_indices]
      threads = [threading.Thread(target=_extract_parallel, args=(used_positive_examples, self.m_positives, dataset, indices[i], indices[i+1], 0, compute_means_and_variances, means, variances), kwargs={'mirror_offset': pos_mirror_offset}) for i in range(self.m_number_of_parallel_threads)]
      [t.start() for t in threads]
      [t.join() for t in threads]

      # negatives
      neg_mirror_offset = len(used_negative_examples) if self.m_mirror_samples else 0
      number_of_indices = len(used_negative_examples)
      indices = [i * number_of_indices / self.m_number_of_parallel_threads for i in range(self.m_number_of_parallel_threads)] + [number_of_indices]
      threads = [threading.Thread(target=_extract_parallel, args=(used_negative_examples, self.m_negatives, dataset, indices[i], indices[i+1], len(used_positive_examples)+pos_mirror_offset), kwargs={'mirror_offset': neg_mirror_offset}) for i in range(self.m_number_of_parallel_threads)]
      [t.start() for t in threads]
      [t.join() for t in threads]
      labels[len(used_positive_examples)+pos_mirror_offset:] = -1.


#    for neg in sorted(used_negative_examples, reverse=True): print neg
    # finally, delete all examples that we returned
    if delete_samples:
      # TODO: implement faster versions of this; currently it is O(n^2)
      for image_index, scale_index, bb_index in sorted(used_positive_examples, reverse=True):
        del self.m_positives[image_index][scale_index][bb_index]
      for image_index, scale_index, bb_index in sorted(used_negative_examples, reverse=True):
        del self.m_negatives[image_index][scale_index][bb_index]

    # return the collected features and labels
    if compute_means_and_variances:
      return (dataset, labels, means, variances)
    else:
      return (dataset, labels)


  def iterate(self, image, feature_extractor, feature_vector):
    """Scales the given image and extracts bounding boxes, computes the features for the given feature extractor and returns an ITERATOR returning a the bounding_box.
    """
    for scale, scaled_image_shape in self._scales(image):
      # prepare the feature extractor to extract features from the given image
      feature_extractor.prepare(image, scale)
      for bb in self._sample(scaled_image_shape):
        # extract features for
        feature_extractor.extract_single(bb, feature_vector)
        yield bb.scale(1./scale)


  def iterate_cascade(self, cascade, image):
    """Iterates over the given image and computes the cascade of classifiers."""

    for scale, scaled_image_shape in self._scales(image):
      # prepare the feature extractor to extract features from the given image
      cascade.prepare(image, scale)
      for bb in self._sample(scaled_image_shape):
        # return bounding box and result
        yield cascade(bb), bb.scale(1./scale)



  def _write(self, name="/scratch/mguenther/temp/examples/image_%i.png", write_positives = True, write_negatives = False):
    """Writes the positive (and negative) training examples to the file with the given file name."""
    if write_positives:
      i = 0
      for image_index, image in enumerate(self.m_images):
        for scale_index, scale in enumerate(self.scales[image_index]):
          if len(self.m_positives[image_index][scale_index]):
            # scale image
            scaled_image = bob.ip.scale(image, scale)

            for bb in self.m_positives[image_index][scale_index]:
              # save part of the
              bob.io.save(scaled_image[bb.top:bb.bottom+1, bb.left:bb.right+1].astype(numpy.uint8), name % i)
              i += 1

