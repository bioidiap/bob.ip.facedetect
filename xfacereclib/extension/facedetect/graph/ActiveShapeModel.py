import bob
import numpy
import facereclib

from .LocalModel import LocalModel

class ActiveShapeModel (LocalModel):

  def __init__(self, patch_size = (96,80), subspace_size = .98, offset = (20,10), local_patch_size = (21,21), local_search_distance = 10):
    LocalModel.__init__(self, patch_size, subspace_size, offset)
    self.local_patch_size = local_patch_size
    self.local_search_distance = local_search_distance

  def extract_patch(self, image, y, x):
    y0 = y - self.local_patch_size[0]/2
    y1 = y0 + self.local_patch_size[0]
    x0 = x - self.local_patch_size[1]/2
    x1 = x0 + self.local_patch_size[1]

    if y0 < 0 or y1 >= image.shape[0] or x0 < 0 or x1 >= image.shape[1]:
      return None

    # extract image patch
    patch = image[y0:y1,x0:x1].flatten().astype(numpy.float)

    # normalize patch to zero mean and unit standard deviation
    return (patch - numpy.mean(patch))/numpy.std(patch)


  def train_local_model(self, train_images, train_annotations):
    """trains the local appearance for each landmark using a small patch around it"""
    facereclib.utils.info("Training local appearance models for %d patches" % len(train_images))
    trainer = bob.trainer.PCATrainer()
    self.local_models = []
    self.local_variances = []
    # iterate over all landmarks
    for a in range(0, train_annotations[0].shape[0], 2):
      facereclib.utils.info("Computing local appearance models for landmark %d" % (a/2))
      local_patches = []
      # extract local patches for all images
      for i in range(len(train_images)):
        y = train_annotations[i][a]
        x = train_annotations[i][a+1]
        patch = self.extract_patch(train_images[i], y, x)
        if patch is not None:
          local_patches.append(patch)

      # train PCA for local patch
      machine, variance = trainer.train(local_patches)
      self.limit_variance(machine, variance)
      self.local_models.append(machine)
      self.local_variances.append(variance)


  def save(self, hdf5):
    LocalModel.save(self, hdf5)
    hdf5.set("LocalPatchSize", self.local_patch_size)
    for a in range(len(self.local_models)):
      hdf5.create_group("Landmark%i"%a)
      hdf5.cd("Landmark%i"%a)
      self.local_models[a].save(hdf5)
      hdf5.set("Variances", self.local_variances[a])
      hdf5.cd("..")

  def load(self, hdf5):
    LocalModel.load(self, hdf5)
    self.local_patch_size = hdf5.read("LocalPatchSize")
    self.local_models = []
    self.local_variances = []
    a = 0
    while hdf5.has_group("Landmark%i"%a):
      hdf5.cd("Landmark%i"%a)
      self.local_models.append(bob.machine.LinearMachine(hdf5))
      self.local_variances.append(hdf5.read("Variances"))
      hdf5.cd("..")
      a += 1


  def init_prediction(self, image):
    # just save reference to image
    self.current_image = image

  def compute_shift(self, prediction):
    # computes the best shift for the current prediction
    shift = []
#    import ipdb; ipdb.set_trace()
    # iterate over all positions
    for a in range(0, len(prediction), 2):
      # get the average position as a starting point
      y = prediction[a]
      x = prediction[a+1]

      # iterate over an area around the local offset
      results = []
      for dy in range(-self.local_search_distance, self.local_search_distance+1):
        for dx in range(-self.local_search_distance, self.local_search_distance+1):
          # extract local patch
          patch = self.extract_patch(self.current_image, y+dy, x+dx)

          if patch is not None:
            """
            # compute mahalanobis distance between given patch and mean
            dist = numpy.linalg.norm(self.local_models[a/2](patch) / numpy.sqrt(self.local_variances[a/2]))
            """

            # compute Euclidean distance between given patch and mean in feature space
            dist = numpy.linalg.norm(self.local_models[a/2](patch))

            """
            # compute distance from feature space
            model = self.local_models[a/2]
            dist = abs(numpy.linalg.norm(patch - model.input_subtract) - numpy.linalg.norm(model(patch)) )
            """

            results.append((dist, dy, dx))

      # get the lowest distance
      if results:
        best_match = min(results, key=lambda x: x[0])
      else:
        best_match = (0,0,0)

      shift.append(best_match[1])
      shift.append(best_match[2])

    return shift

