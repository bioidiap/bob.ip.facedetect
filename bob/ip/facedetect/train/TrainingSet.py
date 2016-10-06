
import bob.io.base
import bob.io.image
import bob.ip.base
import bob.ip.color
import numpy

import os
import collections
import logging
logger = logging.getLogger('bob.ip.facedetect')

from .utils import bounding_box_from_annotation, parallel_part, quasi_random_indices
from .._library import BoundingBox, FeatureExtractor

class TrainingSet:
  """A set of images including bounding boxes that are used as a training set

  The ``TrainingSet`` incorporates information about the data used to train the face detector.
  It is heavily bound to the scripts to re-train the face detector, which are documented in section :ref:`retrain_detector`.

  The training set can be in several stages, which are optimized for speed.
  First, training data is collected in different ways and stored in one or more list files.
  These list files contain the location of the image files, and where the face bounding boxes in the according images are.
  Then, positive and negative features from one or more file lists are extracted and stored in a given ``feature_directory``, where 'positive' features represent faces, and 'negative' features represent the background.
  Finally, the training is performed using only these features only, without keeping track of where they actually stem from.

  **Constructor Documentation**

    Creates an empty training set.

    **Parameters:**

    ``feature_directory`` : str
      The name of a temporary directory, where (intermediate) features will be stored.
      This directory should be able to store several 100GB of data.
  """

  def __init__(self, feature_directory = None):
    self.feature_directory = feature_directory
    self.image_paths = []
    self.bounding_boxes = []

    self.positive_indices = set()
    self.negative_indices = set()

  def add_image(self, image_path, annotations):
    """Adds an image and its bounding boxes to the current list of files

    The bounding boxes are automatically estimated based on the given annotations.

    **Parameters:**

    ``image_path`` : str
      The file name of the image, including its full path

    ``annotations`` : [dict]
      A list of annotations, i.e., where each annotation can be anything that :py:func:`bounding_box_from_annotation` can handle; this list can be empty, in case the image does not contain any faces
    """
    self.image_paths.append(image_path)
    self.bounding_boxes.append([bounding_box_from_annotation(**a) for a in annotations])

  def add_from_db(self, database, files):
    """Adds images and bounding boxes for the given files of a database that follows the :py:ref:`bob.bio.base.database.BioDatabase <bob.bio.base>` interface.

    **Parameters:**

    ``database`` : a derivative of :py:class:`bob.bio.base.database.BioDatabase`
      The database interface, which provides file names and annotations for the given ``files``

    ``files`` : :py:class:`bob.bio.base.database.BioFile` or compatible
      The files (as returned by :py:meth:`bob.bio.base.database.BioDatabase.objects`) which should be added to the training list
    """
    for f in files:
      annotation = database.annotations(f)
      image_path = database.original_file_name(f)
      self.add_image(image_path, [annotation])

  def save(self, list_file):
    """Saves the current list of annotations to the given file.

    **Parameters:**

    ``list_file`` : str
      The name of a list file to write the currently stored list into
    """
    bob.io.base.create_directories_safe(os.path.dirname(list_file))
    with open(list_file, 'w') as f:
      for i in range(len(self.image_paths)):
        f.write(self.image_paths[i])
        for bbx in self.bounding_boxes[i]:
          f.write("\t[%f %f %f %f]" % (bbx.top_f, bbx.left_f, bbx.size_f[0], bbx.size_f[1]))
        f.write("\n")

  def load(self, list_file):
    """Loads the list of annotations from the given file and **appends** it to the current list.

    ``list_file`` : str
      The name of a list file to load and append
    """
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
    """iterate([max_number_of_files]) -> image, bounding_boxes, image_file

    Yields the image and the bounding boxes stored in the training set as an iterator.

    This function loads the images and converts them to gray-scale.
    It yields the image, the list of bounding boxes and the original image file name.

    **Parameters:**

    ``max_number_of_files`` : int or ``None``
      If specified, limit the number of returned data by sub-selection using :py:func:`quasi_random_indices`

    **Yields:**

    ``image`` : array_like(2D)
      The image loaded from file and converted to gray scale

    ``bounding_boxes`` : [:py:class:`BoundingBox`]
      A list of bounding boxes, where faces are found in the image; might be empty (in case of pure background images)

    `` image_file`` : str
      The name of the original image that was read
    """
    indices = quasi_random_indices(len(self), max_number_of_files)
    for index in indices:
      image = bob.io.base.load(self.image_paths[index])
      if len(image.shape) == 3:
        image = bob.ip.color.rgb_to_gray(image)
      # return image and bounding box as iterator
      yield image, self.bounding_boxes[index], self.image_paths[index]


  def _feature_file(self, parallel = None, index = None):
    """Returns the name of an intermediate file for storing features."""
    if index is None:
      index = 0 if parallel is None or "SGE_TASK_ID" not in os.environ else int(os.environ["SGE_TASK_ID"])
    return os.path.join(self.feature_directory, "Features_%02d.hdf5" % index)

  def __len__(self):
    """Returns the number of files stored inside this training set."""
    return len(self.image_paths)


  def extract(self, sampler, feature_extractor, number_of_examples_per_scale = (100, 100), similarity_thresholds = (0.5, 0.8), parallel = None, mirror = False, use_every_nth_negative_scale = 1):
    """Extracts features from **all** images in **all** scales and writes them to file.

    This function iterates over all images that are present in the internally stored list, and extracts features using the given ``feature_extractor`` for every image patch that the given ``sampler`` returns.
    The final features will be stored in the ``feature_directory`` that is set in the constructor.

    For each image, the ``sampler`` samples patch locations, which cover the whole image in different scales.
    For each patch locations is tested, how similar they are to the face bounding boxes that belong to that image, using the Jaccard :py:meth:`BoundingBox.similarity`.
    The similarity is compared to the ``similarity_thresholds``.
    If it is smaller than the first threshold, the patch is considered as background, when it is greater the the second threshold, it is considered as a face, otherwise it is rejected.
    Depending on the image resolution and the number of bounding boxes, this will usually result in some positive and thousands of negative patches per image.
    To limit the total amount of training data, for all scales, only up to a given number of positive and negative patches are kept.
    Also, to further limit the number of negative samples, only every ``use_every_nth_negative_scale`` scale is considered (for the positives, always all scales are processed).

    To increase the number (especially of positive) examples, features can also be extracted for horizontally mirrored images.
    Simply set the ``mirror`` parameter to ``True``.
    Furthermore, this function is designed to be run using several parallel processes, e.g., using the `GridTK <https://pypi.python.org/pypi/gridtk>`_.
    Each of the processes will run on a particular subset of the images, which is defined by the ``SGE_TASK_ID`` environment variable.
    The ``parallel`` parameter defines the total number of parallel processes that are used.

    **Parameters:**

    ``sampler`` : :py:class:`Sampler`
      The sampler to use to sample patches of the images. Please assure that the sampler is set up such that it samples patch locations which can overlap with the face locations.

    ``feature_extractor`` : :py:class:`FeatureExtractor`
      The feature extractor to be used to extract features from image patches

    ``number_of_examples_per_scale`` : (int, int)
      The maximum number of positive and negative examples to extract for each scale of the image

    ``similarity_thresholds`` : (float, float)
      The Jaccard similarity threshold, below which patch locations are considered to be negative, and above which patch locations are considered to be positive examples.

    ``parallel`` : int or ``None``
      If given, the total number of parallel processes, which are used to extract features (the current process index is read from the ``SGE_TASK_ID`` environment variable)

    ``mirror`` : bool
      Extract positive and negative samples also from horizontally mirrored images?

    ``use_every_nth_negative_scale`` : int
      Skip some negative scales to decrease the number of negative examples, i.e., only extract and store negative features, when ``scale_counter % use_every_nth_negative_scale == 0``

      .. note::
         The ``scale_counter`` is not reset between images, so that we might get features from different scales in subsequent images.
    """

    feature_file = self._feature_file(parallel)
    bob.io.base.create_directories_safe(self.feature_directory)

    if parallel is None or "SGE_TASK_ID" not in os.environ or os.environ["SGE_TASK_ID"] == '1':
      extractor_file = os.path.join(self.feature_directory, "Extractor.hdf5")
      hdf5 = bob.io.base.HDF5File(extractor_file, "w")
      feature_extractor.save(hdf5)
      del hdf5

    total_positives, total_negatives = 0, 0

    indices = parallel_part(range(len(self)), parallel)
    if not indices:
      logger.warning("The index range for the current parallel thread is empty.")
    else:
      logger.info("Extracting features for images in range %d - %d of %d", indices[0], indices[-1], len(self))

    hdf5 = bob.io.base.HDF5File(feature_file, "w")
    for index in indices:
      hdf5.create_group("Image-%d" % index)
      hdf5.cd("Image-%d" % index)

      logger.debug("Processing file %d of %d: %s", index+1, indices[-1]+1, self.image_paths[index])

      # load image
      image = bob.io.base.load(self.image_paths[index])
      if image.ndim == 3:
        image = bob.ip.color.rgb_to_gray(image)
      # get ground_truth bounding boxes
      ground_truth = self.bounding_boxes[index]

      # collect image and GT for originally and mirrored image
      images = [image] if not mirror else [image, bob.ip.base.flop(image)]
      ground_truths = [ground_truth] if not mirror else [ground_truth, [gt.mirror_x(image.shape[1]) for gt in ground_truth]]
      parts = "om"

      # now, sample
      scale_counter = -1
      for image, ground_truth, part in zip(images, ground_truths, parts):
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
            hdf5.set("Negatives-%s-%.5f" % (part,scale), negative_features)
            total_negatives += len(negatives)

          # positive features
          if positives:
            positive_features = numpy.zeros((len(positives), feature_extractor.number_of_features), numpy.uint16)
            for i, bb in enumerate(positives):
              feature_extractor.extract_all(bb, positive_features, i)
            hdf5.set("Positives-%s-%.5f" % (part,scale), positive_features)
            total_positives += len(positives)
      # cd backwards after each image
      hdf5.cd("..")

    hdf5.set("TotalPositives", total_positives)
    hdf5.set("TotalNegatives", total_negatives)

  def sample(self, model = None, maximum_number_of_positives = None, maximum_number_of_negatives = None, positive_indices = None, negative_indices = None):
    """sample([model], [maximum_number_of_positives], [maximum_number_of_negatives], [positive_indices], [negative_indices]) -> positives, negatives

    Returns positive and negative samples from the set of positives and negatives.

    This reads the previously extracted feature file (or all of them, in case features were extracted in parallel) and returns features.
    If the ``model`` is not specified, a random sub-selection of positive and negative features is returned.
    When the ``model`` is given, all patches are first classified with the given ``model``, and the ones that are mis-classified most are returned.
    The number of returned positives and negatives can be limited by specifying the ``maximum_number_of_positives`` and ``maximum_number_of_negatives``.

    This function keeps track of the positives and negatives that it once has returned, so it does not return the same positive or negative feature twice.
    However, when you have to restart training from a given point, you can set the ``positive_indices`` and ``negative_indices`` parameters, to retrieve the features for the given indices.
    In this case, no additional features are selected, but the given sets of indices are stored internally.

    .. note::
       The ``positive_indices`` and ``negative_indices`` only have an effect, when ``model`` is ``None``.

    **Parameters:**

    ``model`` : :py:class:`bob.learn.boosting.BoostedMachine` or ``None``
      If given, the ``model`` is used to predict the training features, and the highest mis-predicted features are returned

    ``maximum_number_of_positives, maximum_number_of_negatives`` : int
      The maximum number of positive and negative features to be returned

    ``positive_indices, negative_indices`` : set(int) or ``None``
      The set of positive and negative indices to extract features for, instead of randomly choosing indices; only considered when ``model = None``

    **Returns:**

    ``positives, negatives`` : array_like(2D, uint16)
      The new set of training features for the positive class (faces) and negative class (background).
    """

    # get all existing feature files
    feature_file = self._feature_file(index = 0)
    if os.path.exists(feature_file):
      feature_files = [feature_file]
    else:
      feature_files = []
      i = 1
      while True:
        feature_file = self._feature_file(index = i)
        if not os.path.exists(feature_file):
          break
        feature_files.append(feature_file)
        i += 1

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
        positive_indices = set(quasi_random_indices(positive_count, maximum_number_of_positives))
      if negative_indices is None:
        negative_indices = set(quasi_random_indices(negative_count, maximum_number_of_negatives))
      self.positive_indices |= positive_indices
      self.negative_indices |= negative_indices

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
    """feature_extractor() -> extractor

    Returns the feature extractor used to extract the positive and negative features.

    This feature extractor is stored to file during the :py:meth:`extract` method ran, so this function reads that file (from the ``feature_directory`` set in the constructor) and returns its content.

    **Returns:**

    ``extractor`` : :py:class:`FeatureExtractor`
      The feature extractor used to extract the features stored in the ``feature_directory``
    """
    extractor_file = os.path.join(self.feature_directory, "Extractor.hdf5")
    if not os.path.exists(extractor_file):
      raise IOError("Could not found extractor file %s. Did you already run the extraction process? Did you specify the correct `feature_directory` in the constructor?" % extractor_file)
    hdf5 = bob.io.base.HDF5File(extractor_file)
    return FeatureExtractor(hdf5)
