
import bob.io.base
import bob.io.image
import bob.ip.color
import numpy

import os
import collections
import logging
logger = logging.getLogger('bob.ip.facedetect')

from .utils import bounding_box_from_annotation, parallel_part, quasi_random_indices
from .._library import BoundingBox, FeatureExtractor

class TrainingSet:
  """A set of images including bounding boxes that are used as a training set"""

  def __init__(self, feature_directory = None, list_file=None):
    self.feature_directory = feature_directory
    self.image_paths = []
    self.bounding_boxes = []
    if list_file is not None:
      self.load(list_file)

    self.positive_indices = set()
    self.negative_indices = set()

  def add_image(self, image_path, annotations):
    self.image_paths.append(image_path)
    self.bounding_boxes.append([bounding_box_from_annotation(**a) for a in annotations])

  def add_from_db(self, database, files):
    """Adds image path and bounding boxes from the given annotations."""
    for f in files:
      annotation = database.annotations(f)
      image_path = database.original_file_name(f)
      self.add_image(image_path, [annotation])

  def save(self, list_file):
    """Saves the current list of annotations to the given file."""
    bob.io.base.create_directories_safe(os.path.dirname(list_file))
    with open(list_file, 'w') as f:
      for i in range(len(self.image_paths)):
        f.write(self.image_paths[i])
        for bbx in self.bounding_boxes[i]:
          f.write("\t[%f %f %f %f]" % (bbx.top_f, bbx.left_f, bbx.size_f[0], bbx.size_f[1]))
        f.write("\n")

  def load(self, list_file):
    """Loads the list of annotations from the given file and **appends** it to the current list."""
    with open(list_file) as f:
      for line in f:
        if line and line[0] != '#':
          splits = line.split()
          bounding_boxes = []
          for i in range(1, len(splits), 4):
            assert splits[i][0] == '[' and splits[i+3][-1] == ']'
            bounding_boxes.append(BoundingBox(topleft=(float(splits[i][1:]), float(splits[i+1])), size=(float(splits[i+2]), float(splits[i+3][:-1]))))
          self.image_paths.append(splits[0])
          self.bounding_boxes.append(bounding_boxes)


  def iterate(self, max_number_of_files=None):
    """Returns the image name and the bounding boxes stored in the training set as an iterator."""
    indices = quasi_random_indices(len(self), max_number_of_files)
    for index in indices:
      image = bob.io.base.load(self.image_paths[index])
      if len(image.shape) == 3:
        image = bob.ip.color.rgb_to_gray(image)
      # return image and bounding box as iterator
      yield image, self.bounding_boxes[index], self.image_paths[index]


  def _feature_file(self, parallel = None, index = None):
    if index is None:
      index = '0' if parallel is None or "SGE_TASK_ID" not in os.environ else os.environ["SGE_TASK_ID"]
    if len(index) == 1:
      index = '0' + index
    return os.path.join(self.feature_directory, "Features_%s.hdf5" % index)

  def __len__(self):
    """Returns the number of files stored inside this training set."""
    return len(self.image_paths)


  def extract(self, sampler, feature_extractor, number_of_examples_per_scale = None, similarity_thresholds = (0.5, 0.8), parallel = None, mirror = False, use_every_nth_negative_scale = 1):
    """Extracts **all** features from **all** images in **all** scales and writes them to file.


    similarity_thresholds : (float, float)
      two patches will be compared: the (scaled) annotated patch and the (shifted) extracted patch
      if the similarity is lower than the first value of the similarity_thresholds tuple, it will be accepted as negative example,
      if the similarity is higher than the second value of the similarity_thresholds tuple, it will be accepted as positive example,
      otherwise the patch will be rejected.

    mirror_samples : bool
      extract also mirrored patches
    """

    # TODO: implement mirroring of samples
    if number_of_examples_per_scale is None:
      number_of_examples_per_scale = (None, None)

    feature_file = self._feature_file(parallel)
    bob.io.base.create_directories_safe(self.feature_directory)

    if parallel is None or "SGE_TASK_ID" not in os.environ or os.environ["SGE_TASK_ID"] == '1':
      extractor_file = os.path.join(self.feature_directory, "Extractor.hdf5")
      hdf5 = bob.io.base.HDF5File(extractor_file, "w")
      feature_extractor.save(hdf5)
      del hdf5

    total_positives, total_negatives = 0, 0

    indices = parallel_part(range(len(self.image_paths)), parallel)
    if not indices:
      logger.warning("The index range for the current parallel thread is empty.")
    else:
      logger.info("Extracting features for images in range %d - %d", indices[0], indices[-1])

    hdf5 = bob.io.base.HDF5File(self._feature_file(parallel), "w")
    for index in indices:
      hdf5.create_group("Image-%d" % index)
      hdf5.cd("Image-%d" % index)

      logger.debug("Processing file %d of %d: %s", index, indices[-1], self.image_paths[index])

      # load image
      image = bob.io.base.load(self.image_paths[index])
      if len(image.shape) == 3:
        image = bob.ip.color.rgb_to_gray(image)
      # get ground_truth bounding boxes
      ground_truth = self.bounding_boxes[index]

      # collect image and GT for originally and mirrored image
      images = [image] if not mirror else [image, image[:,::-1].copy()]
      ground_truths = [ground_truth] if not mirror else [ground_truth, [gt.mirror_x(image.shape[1]) for gt in ground_truth]]

      # now, sample
      scale_counter = -1
      for image, ground_truth in zip(images, ground_truths):
        for scale, scaled_image_shape in sampler.scales(image):
          scale_counter += 1
          scaled_gt = [gt.scale(scale) for gt in ground_truth]
          positives = []
          negatives = []
          # iterate over all possible positions in the image
          for bb in sampler.sample_scaled(scaled_image_shape):
            # check if the patch is a positive example
            positive = False
            negative = True
            for gt in scaled_gt:
              similarity = bb.similarity(gt)
              if similarity > similarity_thresholds[1]:
                positive = True
                break
              if similarity > similarity_thresholds[0]:
                negative = False
                break

            if positive:
              positives.append(bb)
            elif negative and scale_counter % use_every_nth_negative_scale == 0:
              negatives.append(bb)

          # per scale, limit the number of positive and negative samples
          positives = [positives[i] for i in quasi_random_indices(len(positives), number_of_examples_per_scale[0])]
          negatives = [negatives[i] for i in quasi_random_indices(len(negatives), number_of_examples_per_scale[1])]

          # extract features
          feature_extractor.prepare(image, scale)
          # .. negative features
          if negatives:
            negative_features = numpy.zeros((len(negatives), feature_extractor.number_of_features), numpy.uint16)
            for i, bb in enumerate(negatives):
              feature_extractor.extract_all(bb, negative_features, i)
            hdf5.set("Negatives-%.5f" % scale, negative_features)
            total_negatives += len(negatives)

          # positive features
          if positives:
            positive_features = numpy.zeros((len(positives), feature_extractor.number_of_features), numpy.uint16)
            for i, bb in enumerate(positives):
              feature_extractor.extract_all(bb, positive_features, i)
            hdf5.set("Positives-%.5f" % scale, positive_features)
            total_positives += len(positives)
        # cd backwards after each (mirrored) image
        hdf5.cd("..")

    hdf5.set("TotalPositives", total_positives)
    hdf5.set("TotalNegatives", total_negatives)

  def sample(self, model = None, maximum_number_of_positives = None, maximum_number_of_negatives = None, positive_indices = None, negative_indices = None):
    """Returns positive and negative samples from the set of positives and negatives."""

    # get all existing feature files
    feature_file = self._feature_file(index = '0')
    if os.path.exists(feature_file):
      feature_files = [feature_file]
    else:
      feature_files = []
      i = 1
      feature_file = self._feature_file(index = str(i))
      while os.path.exists(feature_file):
        feature_files.append(feature_file)
        i += 1
        feature_file = self._feature_file(index = str(i))

    features = []
    labels = []

    # make a first iteration through the feature files and count the number of positives and negatives
    positive_count, negative_count = 0, 0
    logger.info("Reading %d feature files", len(feature_files))
    for feature_file in feature_files:
      logger.debug(".. Loading file %s", feature_file)
      hdf5 = bob.io.base.HDF5File(feature_file)
      positive_count += hdf5.get("TotalPositives")
      negative_count += hdf5.get("TotalNegatives")
      del hdf5

    if model is None:
      # get a list of indices and store them, so that we don't re-use them next time
      if positive_indices is None:
        positive_indices = quasi_random_indices(positive_count, maximum_number_of_positives)
      if negative_indices is None:
        negative_indices = quasi_random_indices(negative_count, maximum_number_of_negatives)
      self.positive_indices |= set(positive_indices)
      self.negative_indices |= set(negative_indices)

      # now, iterate through the files again and sample
      positive_indices = collections.deque(sorted(positive_indices))
      negative_indices = collections.deque(sorted(negative_indices))

      logger.info("Extracting %d of %d positive and %d of %d negative samples" % (len(positive_indices), positive_count, len(negative_indices), negative_count))

      positive_count, negative_count = 0, 0
      for feature_file in feature_files:
        hdf5 = bob.io.base.HDF5File(feature_file)
        for image in sorted(hdf5.sub_groups(recursive=False, relative=True)):
          hdf5.cd(image)
          for scale in sorted(hdf5.keys(relative=True)):
            read = hdf5.get(scale)
            size = read.shape[0]
            if scale.startswith("Positives"):
              # copy positive data
              while positive_indices and positive_count <= positive_indices[0] and positive_count + size > positive_indices[0]:
                assert positive_indices[0] >= positive_count
                features.append(read[positive_indices.popleft() - positive_count, :])
                labels.append(1)
              positive_count += size
            else:
              # copy negative data
              while negative_indices and negative_count <= negative_indices[0] and negative_count + size > negative_indices[0]:
                assert negative_indices[0] >= negative_count
                features.append(read[negative_indices.popleft() - negative_count, :])
                labels.append(-1)
              negative_count += size
          hdf5.cd("..")
      # return features and labels
      return numpy.array(features), numpy.array(labels)

    else:
      positive_count -= len(self.positive_indices)
      negative_count -= len(self.negative_indices)
      logger.info("Getting worst %d of %d positive and worst %d of %d negative examples", min(maximum_number_of_positives, positive_count), positive_count, min(maximum_number_of_negatives, negative_count), negative_count)

      # compute the worst features based on the current model
      worst_positives, worst_negatives = [], []
      positive_count, negative_count = 0, 0

      for feature_file in feature_files:
        hdf5 = bob.io.base.HDF5File(feature_file)
        for image in sorted(hdf5.sub_groups(recursive=False, relative=True)):
          hdf5.cd(image)
          for scale in sorted(hdf5.keys(relative=True)):
            read = hdf5.get(scale)
            size = read.shape[0]
            prediction = bob.blitz.array((size,), numpy.float64)
            # forward features through the model
            result = model.forward(read, prediction)
            if scale.startswith("Positives"):
              indices = [i for i in range(size) if positive_count + i not in self.positive_indices]
              worst_positives.extend([(prediction[i], positive_count + i, read[i]) for i in indices if prediction[i] <= 0])
              positive_count += size
            else:
              indices = [i for i in range(size) if negative_count + i not in self.negative_indices]
              worst_negatives.extend([(prediction[i], negative_count + i, read[i]) for i in indices if prediction[i] >= 0])
              negative_count += size
          hdf5.cd("..")

        # cut off good results
        if maximum_number_of_positives is not None and len(worst_positives) > maximum_number_of_positives:
          # keep only the positives with the low predictions (i.e., the worst)
          worst_positives = sorted(worst_positives, key=lambda k: k[0])[:maximum_number_of_positives]
        if maximum_number_of_negatives is not None and len(worst_negatives) > maximum_number_of_negatives:
          # keep only the negatives with the high predictions (i.e., the worst)
          worst_negatives = sorted(worst_negatives, reverse=True, key=lambda k: k[0])[:maximum_number_of_negatives]

      # mark all indices to be used
      self.positive_indices |= set(k[1] for k in worst_positives)
      self.negative_indices |= set(k[1] for k in worst_negatives)

      # finally, collect features and labels
      return numpy.array([f[2] for f in worst_positives] + [f[2] for f in worst_negatives]), numpy.array([1]*len(worst_positives) + [-1]*len(worst_negatives))


  def feature_extractor(self):
    """Returns the feature extractor used to extract the positive and negative features."""
    extractor_file = os.path.join(self.feature_directory, "Extractor.hdf5")
    hdf5 = bob.io.base.HDF5File(extractor_file)
    return FeatureExtractor(hdf5)
