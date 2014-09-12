
import math
import numpy
import facereclib
import itertools

from .._features import BoundingBox

import bob.ip.base

import threading


class Sampler:
  """This class generates and contains bounding boxes positive and negative examples used for face detection."""

  def __init__(self, patch_size = (24,20), scale_factor = math.pow(2., -1./16.), lowest_scale = math.pow(2., -6.), distance = 2, similarity_thresholds = (0.3, 0.7), mirror_samples=False, number_of_parallel_threads=1):
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
    self.m_targets = []

    self.m_patch_size = patch_size
    self.m_patch_box = BoundingBox((0, 0), patch_size)
    self.m_scale_factor = scale_factor
    self.m_lowest_scale = lowest_scale
    self.m_distance = distance
    self.m_similarity_thresholds = similarity_thresholds
    self.m_number_of_parallel_threads = number_of_parallel_threads
    self.m_mirror_samples = mirror_samples
    self.m_target_length = 0


  def scale_self(self, scale):
    patch_size = (int(round(self.m_patch_box.size[0]*scale)), int(round(self.m_patch_box.height[1] * scale)))
    sampler = Sampler(patch_size, scale_factor=self.m_scale_factor, lowest_scale=self.m_lowest_scale, distance=int(math.ceil(self.m_distance * scale)), similarity_thresholds=self.m_similarity_thresholds, mirror_samples=self.m_mirror_samples, number_of_parallel_threads=self.m_number_of_parallel_threads)

    sampler.m_scales = [[s * scale for s in scales] for scales in self.m_scales]
    sampler.m_images = self.m_images
    sampler.m_positives = [[[b.scale(scale) for b in bb] for bb in bbs] for bbs in self.m_positives]
    sampler.m_negatives = [[[b.scale(scale) for b in bb] for bb in bbs] for bbs in self.m_negatives]
    sampler.m_targets = [[[t * scale for t in tar] for tar in target] for target in self.m_targets]
    sampler.m_target_length = self.m_target_length

    return sampler



  def _scales(self, image):
    minimum_scale = max(self.m_patch_box.size_f[0] / image.shape[0], self.m_patch_box.size_f[1] / image.shape[1])
    if self.m_lowest_scale:
      maximum_scale = min(minimum_scale / self.m_lowest_scale, 1.)
    else:
      maximum_scale = 1.
    current_scale_power = 0.

    while True:
      # scale the image
      scale = minimum_scale * math.pow(self.m_scale_factor, current_scale_power)
      if scale > maximum_scale:
        # image is smaller than the requested minimum size
        break
      current_scale_power -= 1.
      scaled_image_shape = bob.ip.base.scaled_output_shape(image, scale)

      yield scale, scaled_image_shape


  def _sample(self, scaled_image_shape):
    """Returns an iterator that iterates over the sampled positions in the image."""
    for y in range(0, scaled_image_shape[0]-self.m_patch_box.bottomright[0]-1, self.m_distance):
      for x in range(0, scaled_image_shape[1]-self.m_patch_box.bottomright[1]-1, self.m_distance):
        # create bounding box for the image
        yield self.m_patch_box.shift((y,x))



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


  def add_targets(self, image, bounding_boxes, annotations, number_of_samples_per_scale = None, annotation_types = ['reye', 'leye']):
    """Adds examples from the given image, using the given ground_truth bounding boxes and the given ."""

    def _targets(bb, ann):
      # extracts the targets for the current bounding box from the given annotations
      target = numpy.ndarray((self.m_target_length,), numpy.float64)
      for i, t in enumerate(annotation_types):
        # we define the annotation positions relative to the center of the bounding box
        # This is different to Cosmins implementation, who used the top-left corner instead of the center
        target[2*i] = (ann[i][0] - bb.center[0])
        target[2*i+1] = (ann[i][1] - bb.center[1])
      return target

    # assure that the eye positions are the first two annotations (required by the Jesorsky error measure)
    assert 'reye' in annotation_types[:2] and 'leye' in annotation_types[:2]
    # remove the annotations that are incomplete
    for i in range(len(annotations)-1, 0, -1):
      for t in annotation_types:
        if t not in annoations[i]:
          facereclib.utils.warning("Removing bounding box since annotations are incomplete.")
          del annotations[i]
          del bounding_boxes[i]
          break

    self.m_target_length = 2*len(annotation_types)

    # remeber the image
    self.m_images.append(image)
    # remember the possible scales for this image
    self.m_scales.append([])
    self.m_targets.append([])
    self.m_positives.append([])

    for scale, scaled_image_shape in self._scales(image):

      scaled_gt = [bb.scale(scale) for bb in bounding_boxes]
      scaled_ann = [[(a[t][0]*scale, a[t][1]*scale) for t in annotation_types] for a in annotations]
      positives = []
      targets = []

      # iterate over all possible positions in the image
      for bb in self._sample(scaled_image_shape):
        # check if the patch is a positive example
        for i, gt in enumerate(scaled_gt):
          similarity = bb.similarity(gt)
          if similarity > self.m_similarity_thresholds[1]:
            positives.append(bb)
            targets.append(_targets(bb, scaled_ann[i]))

      # at the end, add found patches
      if len(targets):
        self.m_scales[-1].append(scale)
        self.m_positives[-1].append([positives[i] for i in facereclib.utils.quasi_random_indices(len(positives), number_of_samples_per_scale)])
        self.m_targets[-1].append([targets[i] for i in facereclib.utils.quasi_random_indices(len(targets), number_of_samples_per_scale)])


  def _get_all(self, pos_or_neg):
    for image_count in range(len(pos_or_neg)):
      for scale_count in range(len(pos_or_neg[image_count])):
        for bb_count in range(len(pos_or_neg[image_count][scale_count])):
          yield (image_count, scale_count, bb_count)


  def get(self, feature_extractor, model = None, maximum_number_of_positives = None, maximum_number_of_negatives = None, delete_samples = False, compute_means_and_variances = False, loss_function = None):

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
            fex.extract_indexed(bb, feature_vector)
            # compute the current prediction of the model
            if self.m_target_length != 0:
              scores = numpy.ndarray(self.m_target_length)
              model.forward_p(feature_vector, scores)
              pos.append((scores, self.m_targets[image_index][scale_index][bb_index], image_index, scale_index, bb_index))
            else:
              pos.append((model.forward_p(feature_vector), image_index, scale_index, bb_index))
          if self.m_target_length == 0:
            for bb_index, bb in enumerate(self.m_negatives[image_index][scale_index]):
              # extract the features for the current bounding box
              fex.extract_indexed(bb, feature_vector)
              # compute the current prediction of the model
              neg.append((model.forward_p(feature_vector), image_index, scale_index, bb_index))


    def _extract_parallel(examples, bounding_boxes, dataset, first, last, offset, compute_means = False, means=[], variances=[], mirror_offset = 0, labels = None):
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
        fex.extract_all(bb, dataset, index + offset)
        if self.m_target_length != 0:
          labels[index+offset] = self.m_targets[image_index][scale_index][bb_index]
        if mirror_offset:
          mex.extract_all(bb.mirror_x(mex.image.shape[1]), dataset, index + offset + mirror_offset)
          if self.m_target_length != 0:
            raise NotImplementedError("Using mirrored regression data is not supported (yet).")
            labels[index+offset+mirror_offset] = [b if a % 2 == 0 else -b for (a,b) in enumerate(self.m_targets[image_index][scale_index][bb_index])]
        if compute_means:
          m,v = fex.mean_and_variance(bb)
          means[index+offset] = m
          variances[index+offset] = v

    """Returns a pair of features and labels that can be used for training, after extracting the features using the given feature extractor.
    If number_of_positives and/or number_of_negatives are given, the number of examples is limited to these numbers."""

    # get the maximum number of examples
    pos_count = len(list(self._get_all(self.m_positives)))
    neg_count = len(list(self._get_all(self.m_negatives)))
    if self.m_mirror_samples:
      pos_count *= 2
      neg_count *= 2
    num_pos = pos_count if maximum_number_of_positives is None else min(maximum_number_of_positives, pos_count)
    num_neg = neg_count if maximum_number_of_negatives is None else min(maximum_number_of_negatives, neg_count)


    # create feature as required for training
    dataset = numpy.ndarray((num_pos + num_neg, feature_extractor.number_of_features), numpy.uint16)
    if self.m_target_length:
      labels = numpy.ndarray((num_pos + num_neg, self.m_target_length), numpy.float64)
    else:
      labels = numpy.ones((num_pos + num_neg, ), numpy.float64)
    means = numpy.ndarray((num_pos/2 if self.m_mirror_samples else num_pos, ), numpy.float64)
    variances = numpy.ndarray((num_pos/2 if self.m_mirror_samples else num_pos, ), numpy.float64)


    # get the positive and negative examples
    if model is None:
      # collect positive and negative examples
      all_positive_examples = list(self._get_all(self.m_positives))
      all_negative_examples = list(self._get_all(self.m_negatives))

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
              feature_extractor.extract_indexed(bb, feature_vector)
              if self.m_target_length != 0:
                scores = numpy.ndarray(self.m_target_length)
                model.forward_p(feature_vector, scores)
                positive_values.append((scores, self.m_targets[image_index][scale_index][bb_index], image_index, scale_index, bb_index))
              else:
                positive_values.append((model.forward_p(feature_vector), image_index, scale_index, bb_index))

            if self.m_target_length == 0:
              for bb_index, bb in enumerate(self.m_negatives[image_index][scale_index]):
                # extract the features for the current bounding box
                feature_extractor.extract_indexed(bb, feature_vector)
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

      # in case of multi-variate regression, the real error has to be computed:
      if self.m_target_length != 0 and len(positive_values) != 0:
        assert loss_function is not None
        # compute the loss for the positive examples
        targets = numpy.vstack([p[1] for p in positive_values])
        scores = numpy.vstack([p[0] for p in positive_values])
        errors = loss_function.loss(targets, scores)
        positive_values = [[-errors[i,0]] + list(positive_values[i][2:]) for i in range(len(positive_values))]

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
#        print "extracted features", i, bb, self.m_images[image_index].shape, self.m_scales[image_index][scale_index]
        try:
          feature_extractor.extract_all(bb, dataset, i)
        except Exception as e:
          print bb, self.m_scales[image_index][scale_index], [s*self.m_scales[image_index][scale_index] for s in self.m_images[image_index].shape]
          raise
        if self.m_target_length != 0:
          labels[i] = self.m_targets[image_index][scale_index][bb_index]
        if self.m_mirror_samples:
          mirror_extractor.extract_all(bb.mirror_x(mirror_extractor.image.shape[1]), dataset, i + mirror_offset)
          if self.m_target_length != 0:
            raise NotImplementedError("Using mirrored regression data is not supported (yet).")
            labels[i + mirror_offset] = [b if a % 2 == 0 else -b for (a,b) in enumerate(self.m_targets[image_index][scale_index][bb_index])]
        if compute_means_and_variances:
          m,v = feature_extractor.mean_variance(bb, True)
          if m == 0 or v == 0:
            facereclib.utils.warn("In image number %d of %d there was a positive bounding box containing mean %f and variance %f" % (image_index, len(self.m_images), m, v))
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
        feature_extractor.extract_all(bb, dataset, i)
        labels[i] = -1.
        if self.m_mirror_samples:
          mirror_extractor.extract_all(bb.mirror_x(mirror_extractor.image.shape[1]), dataset, i + mirror_offset)
          labels[i + mirror_offset] = -1.
        i += 1
    else: # parallel implementation
      # positives
      pos_mirror_offset = len(used_positive_examples) if self.m_mirror_samples else 0
      number_of_indices = len(used_positive_examples)
      indices = [i * number_of_indices / self.m_number_of_parallel_threads for i in range(self.m_number_of_parallel_threads)] + [number_of_indices]
      threads = [threading.Thread(target=_extract_parallel, args=(used_positive_examples, self.m_positives, dataset, indices[i], indices[i+1], 0, compute_means_and_variances, means, variances), kwargs={'mirror_offset': pos_mirror_offset, 'labels':labels}) for i in range(self.m_number_of_parallel_threads)]
      [t.start() for t in threads]
      [t.join() for t in threads]

      # negatives
      if self.m_target_length == 0:
        neg_mirror_offset = len(used_negative_examples) if self.m_mirror_samples else 0
        number_of_indices = len(used_negative_examples)
        indices = [i * number_of_indices / self.m_number_of_parallel_threads for i in range(self.m_number_of_parallel_threads)] + [number_of_indices]
        threads = [threading.Thread(target=_extract_parallel, args=(used_negative_examples, self.m_negatives, dataset, indices[i], indices[i+1], len(used_positive_examples)+pos_mirror_offset), kwargs={'mirror_offset': neg_mirror_offset}) for i in range(self.m_number_of_parallel_threads)]
        [t.start() for t in threads]
        [t.join() for t in threads]
        labels[len(used_positive_examples)+pos_mirror_offset:] = -1.

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


  def get_images_and_annotations(self, maximum_number_of_images = None):
    # compute elements that should be returned
    all_examples = list(self._get_all(self.m_positives))
    num_images = len(all_examples)
    if self.m_mirror_samples:
      raise NotImplementedError("Using mirrored regression data is not supported (yet).")
      num_images *= 2
    num_images = num_images if maximum_number_of_images is None else min(maximum_number_of_images, num_images)

    # collect positive and negative examples
    # simply compute a random subset of both lists
    # (for now, this is quasi-random)
    used_examples = [all_examples[i] for i in facereclib.utils.quasi_random_indices(len(all_examples), num_images/2 if self.m_mirror_samples else num_images)]

    images = []
    annotations = []
    # return images and annotations for the sub-selection

    last_image_index = -1
    last_scale_index = -1
    i = 0
    # append positive examples
    for image_index, scale_index, bb_index in used_examples:
      # prepare for this image, if it has changed
      if last_scale_index != scale_index or last_image_index != image_index:
        last_scale_index = scale_index
        last_image_index = image_index
        scaled_image = bob.ip.base.scale(self.m_images[image_index], self.m_scales[image_index][scale_index])

      # extract and append features
      bb = self.m_positives[image_index][scale_index][bb_index]
      images.append(scaled_image[bb.topleft[0]:bb.bottomright[0]+1, bb.topleft[1]:bb.bottomright[1]+1].copy())
      annotations.append(self.m_targets[image_index][scale_index][bb_index])

    return images, annotations


  def iterate(self, image, feature_extractor, feature_vector):
    """Scales the given image and extracts bounding boxes, computes the features for the given feature extractor and returns an ITERATOR returning a the bounding_box.
    """
    for scale, scaled_image_shape in self._scales(image):
      # prepare the feature extractor to extract features from the given image
      feature_extractor.prepare(image, scale)
      for bb in self._sample(scaled_image_shape):
        # extract features for
        feature_extractor.extract_indexed(bb, feature_vector)
        yield bb.scale(1./scale)


  def iterate_cascade(self, cascade, image):
    """Iterates over the given image and computes the cascade of classifiers."""

    for scale, scaled_image_shape in self._scales(image):
      # prepare the feature extractor to extract features from the given image
      cascade.prepare(image, scale)
      for bb in self._sample(scaled_image_shape):
        # return bounding box and result
        yield cascade(bb), bb.scale(1./scale)


