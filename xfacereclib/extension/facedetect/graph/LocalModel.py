import facereclib
import bob
import numpy


def display(image, annotations, color=(255,0,0), sleep=1):
  import matplotlib.pyplot
  import time

  colored = bob.ip.gray_to_rgb(image).astype(numpy.uint8)
  for y,x in [annotations[i:i+2] for i in range(0, len(annotations), 2)]:
    bob.ip.draw_cross(colored, y=int(y), x=int(x), radius=5, color=color)

  matplotlib.pyplot.clf()
  matplotlib.pyplot.imshow(numpy.rollaxis(numpy.rollaxis(colored, 2),2))
  matplotlib.pyplot.draw()
  time.sleep(sleep)



class LocalModel:

  def __init__(self, patch_size = (96,80), subspace_size = .98, offset = (0,0)):
    self.patch_size = patch_size
    self.offset = offset
    self.subspace_size = subspace_size


  def get_average(self, bb):
    scale = float(bb.height_f) / float(self.patch_size[0])
    offset = [bb.top_f - self.offset[0]*scale, bb.left_f - self.offset[1]*scale] * (len(self.shape_model.input_subtract)/2)

    return [self.shape_model.input_subtract[2*a+b] * scale + offset[2*a+b] for a in range(len(self.shape_model.input_subtract)/2) for b in (0,1)]


  def limit_variance(self, model, variances):
    cummulated = numpy.cumsum(variances) / numpy.sum(variances)
    for index in range(len(cummulated)):
      if cummulated[index] > self.subspace_size:
        break
    model.resize(model.shape[0], index)
    variances.resize(index, refcheck=False)


  def train_shape_model(self, annotations):
    # compute shape model
    facereclib.utils.info("Computing shape model from %d patches" % len(annotations))
    pca = bob.trainer.PCATrainer()
    self.shape_model, variances = pca.train(annotations)

    self.limit_variance(self.shape_model, variances)
    # limit subspace size
    self.shape_standard_deviation = numpy.sqrt(variances)


  def train(self, train_images, train_annotations):
    # train shape model
    self.train_shape_model(train_annotations)

    # train the local apperance model (using derived class methods)
    self.train_local_model(train_images, train_annotations)

  def train_local_model(images, annotations):
    pass

  def number_of_predictions(self):
    return self.shape_model.input_subtract.shape[0]

  def init_prediction(self, image):
    pass

  def predict(self, image, bounding_box, max_iterations = 10):
    # extract image with offset boundaries
    t,b,l,r = bounding_box.top-self.offset[0], bounding_box.bottom+self.offset[0]+1, bounding_box.left-self.offset[1], bounding_box.right+self.offset[1]+1
    if t < 0 or b >= image.shape[0] or l < 0 or r > image.shape[1]:
      return None
    image = image[t:b, l:r]
    # initialize the prediction (in derived class)
    self.init_prediction(image)

    # start with average shape model
    prediction = self.shape_model.input_subtract
    last_prediction = prediction.copy()

#    import ipdb; ipdb.set_trace()

    # refine prediction
    for i in range(max_iterations):
      # remove outliers with active shape model
      y = self.shape_model.forward(numpy.array(prediction))
      # limit the shape to maximum 3 times the standard deviations
      limit = 3 * self.shape_standard_deviation
      numpy.clip(y, -limit, limit, y)
      prediction = numpy.around(numpy.inner(self.shape_model.weights, y) + self.shape_model.input_subtract)

      # update the prediction
      shift = self.compute_shift(prediction)
      prediction += shift

      # check for convergence
      if numpy.all(numpy.abs(shift) < 0.5) or numpy.allclose(prediction, last_prediction, atol = 0.5):
        break
      last_prediction = prediction.copy()


    for i in range(0, len(prediction), 2):
      prediction[i] -= self.offset[0]
      prediction[i+1] -= self.offset[1]

    return prediction


  def save(self, hdf5):
    hdf5.set("Offset", self.offset)
    hdf5.set("PatchSize", self.patch_size)
    hdf5.create_group("ShapeModel")
    hdf5.cd("ShapeModel")
    self.shape_model.save(hdf5)
    hdf5.set("StandardDeviation", self.shape_standard_deviation)
    hdf5.cd("..")

  def load(self, hdf5):
    self.offset = hdf5.read("Offset")
    self.patch_size = hdf5.read("PatchSize")
    hdf5.cd("ShapeModel")
    self.shape_model = bob.machine.LinearMachine(hdf5)
    self.shape_standard_deviation = hdf5.read("StandardDeviation")
    hdf5.cd("..")


