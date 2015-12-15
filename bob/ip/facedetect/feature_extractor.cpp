/**
 * @author Manuel Guenther <manuel.guenther@idiap.ch>
 * @date Wed Jul  2 14:38:18 CEST 2014
 *
 * @brief Binds the DCTFeatures class to python
 *
 * Copyright (C) 2011-2014 Idiap Research Institute, Martigny, Switzerland
 */

#include "main.h"
#include <boost/format.hpp>

/******************************************************************/
/************ Constructor Section *********************************/
/******************************************************************/

static auto FeatureExtractor_doc = bob::extension::ClassDoc(
  BOB_EXT_MODULE_PREFIX ".FeatureExtractor",
  "This class extracts LBP features of several types from a given image patch of a certain size",
  "LBP features are extracted using different variants of :py:class:`bob.ip.base.LBP` feature extractors. "
  "All LBP features of one patch are stored in a single long feature vector of type :py:class:`numpy.uint16`."
).add_constructor(
  bob::extension::FunctionDoc(
    "__init__",
    "Generates a new feature extractor for the given ``patch_size`` using one or several feature extractors",
    "The constructor can be called in different ways:\n\n"
    "* The first constructor initializes a feature extractor with no LBP extractor. "
    "Please use the :py:meth:`append` function to add LBP extractors.\n"
    "* In the second constructor, a given list of LBP extractors is specified.\n"
    "* The third constructor initializes a tight set of LBP extractors for different :py:attr:`bob.ip.base.LBP.radii`, by adding all possible combinations of x- and y- radii, until the ``patch_size`` is too small, or ``min_size`` (start) or ``max_size`` (end) is reached.\n"
    "* The fourth constructor copies all LBP extractors from the given :py:class:`FeatureExtractor`\n"
    "* The last constructor read the configuration from the given :py:class:`bob.io.base.HDF5File`.",
    true
  )
  .add_prototype("patch_size", "")
  .add_prototype("patch_size, extractors", "")
  .add_prototype("patch_size, template, [overlap], [square], [min_size], [max_size]", "")
  .add_prototype("other", "")
  .add_prototype("hdf5", "")
  .add_parameter("patch_size", "(int, int)", "The size of the patch to extract from the images")
  .add_parameter("extractors", "[:py:class:`bob.ip.base.LBP`]", "The LBP classes to use as extractors")
  .add_parameter("template", ":py:class:`bob.ip.base.LBP`", "The LBP classes to use as template for all extractors")
  .add_parameter("overlap", "bool", "[default: False] Should overlapping LBPs be created?")
  .add_parameter("square", "bool", "[default: False] Should only square LBPs be created?")
  .add_parameter("min_size", "int", "[default: 1] The minimum radius of LBP")
  .add_parameter("max_size", "int", "[default: MAX_INT] The maximum radius of LBP (limited by patch size)")
  .add_parameter("other", ":py:class:`FeatureExtractor`", "The feature extractor to use for copy-construction")
  .add_parameter("hdf5", ":py:class:`bob.io.base.HDF5File`", "The HDF5 file to read the extractors from")
);


static int PyBobIpFacedetectFeatureExtractor_init(PyBobIpFacedetectFeatureExtractorObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY

  char** kwlist1 = FeatureExtractor_doc.kwlist(0);
  char** kwlist2 = FeatureExtractor_doc.kwlist(1);
  char** kwlist3 = FeatureExtractor_doc.kwlist(2);

  // get the number of command line arguments
  Py_ssize_t nargs = (args?PyTuple_Size(args):0) + (kwargs?PyDict_Size(kwargs):0);

  blitz::TinyVector<int,2> patch_size;
  if (nargs == 1){
    // get first arguments
    PyObject* first = PyTuple_Size(args) ? PyTuple_GET_ITEM(args, 0) : PyList_GET_ITEM(make_safe(PyDict_Values(kwargs)).get(), 0);
    if (PyBobIoHDF5File_Check(first)){
      self->cxx.reset(new bob::ip::facedetect::FeatureExtractor(*((PyBobIoHDF5FileObject*)first)->f));
      return 0;
    }
    if (PyBobIpFacedetectFeatureExtractor_Check(first)){
      self->cxx.reset(new bob::ip::facedetect::FeatureExtractor(*((PyBobIpFacedetectFeatureExtractorObject*)first)->cxx));
      return 0;
    }
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "(ii)", kwlist1, &patch_size[0], &patch_size[1])){
      return -1;
    }
    self->cxx.reset(new bob::ip::facedetect::FeatureExtractor(patch_size));
    return 0;
  }

  // more than one arg
  PyObject* list;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "(ii)O!", kwlist2, &patch_size[0], &patch_size[1], &PyList_Type, &list)){
    PyErr_Clear();
    // try the third option
    PyBobIpBaseLBPObject* lbp;
    PyObject* overlap = 0,* square = 0;
    int min_size = 1, max_size = std::numeric_limits<int>::max();

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "(ii)O&|O!O!ii", kwlist3, &patch_size[0], &patch_size[1], &PyBobIpBaseLBP_Converter, &lbp, &PyBool_Type, &overlap, &PyBool_Type, &square, &min_size, &max_size)) return -1;
    auto lbp_ = make_safe(lbp);
    self->cxx.reset(new bob::ip::facedetect::FeatureExtractor(patch_size, *lbp->cxx, f(overlap), f(square), min_size, max_size));
    return 0;
  }
  // with list of LBP's
  std::vector<boost::shared_ptr<bob::ip::base::LBP>> lbps(PyList_GET_SIZE(list));
  for (Py_ssize_t i = 0; i < PyList_GET_SIZE(list); ++i){
    PyObject* lbp = PyList_GET_ITEM(list, i);
    if (!PyBobIpBaseLBP_Check(lbp)){
      PyErr_Format(PyExc_TypeError, "%s : expected a list of LBP objects, but object number %d is of type `%s'", Py_TYPE(self)->tp_name, (int)i, Py_TYPE(lbp)->tp_name);
      return -1;
    }
    lbps[i] = ((PyBobIpBaseLBPObject*)lbp)->cxx;
  }
  self->cxx.reset(new bob::ip::facedetect::FeatureExtractor(patch_size, lbps));
  return 0;
  BOB_CATCH_MEMBER("cannot create FeatureExtractor", -1)
}

static void PyBobIpFacedetectFeatureExtractor_delete(PyBobIpFacedetectFeatureExtractorObject* self) {
  self->cxx.reset();
  Py_TYPE(self)->tp_free((PyObject*)self);
}

int PyBobIpFacedetectFeatureExtractor_Check(PyObject* o) {
  return PyObject_IsInstance(o, reinterpret_cast<PyObject*>(&PyBobIpFacedetectFeatureExtractor_Type));
}

#if 0 // TODO:
static PyObject* FeatureExtractor_RichCompare(FeatureExtractorObject* self, PyObject* other, int op) {
  BOB_TRY
  if (!FeatureExtractor_Check(other)) {
    PyErr_Format(PyExc_TypeError, "cannot compare `%s' with `%s'", Py_TYPE(self)->tp_name, Py_TYPE(other)->tp_name);
    return 0;
  }
  auto other_ = reinterpret_cast<FeatureExtractorObject*>(other);
  switch (op) {
    case Py_EQ:
      if (*self->cxx==*other_->cxx) Py_RETURN_TRUE; else Py_RETURN_FALSE;
    case Py_NE:
      if (*self->cxx==*other_->cxx) Py_RETURN_FALSE; else Py_RETURN_TRUE;
    default:
      Py_INCREF(Py_NotImplemented);
      return Py_NotImplemented;
  }
  BOB_CATCH_MEMBER("cannot compare FeatureExtractor objects", 0)
}

PyObject* FeatureExtractor_Str(FeatureExtractorObject* self) {
  BOB_TRY
  return PyString_FromString((boost::format("<BB topleft=(%3.2d,%3.2d), bottomright=(%3.2d,%3.2d)>") % self->cxx->top() % self->cxx->left() % self->cxx->bottom() % self->cxx->right()).str().c_str());
  BOB_CATCH_MEMBER("cannot create __repr__ string", 0)
}
#endif


/******************************************************************/
/************ Variables Section ***********************************/
/******************************************************************/

static auto image = bob::extension::VariableDoc(
  "image",
  "array_like <2D, uint8>",
  "The (prepared) image the next features will be extracted from, read access only"
);
PyObject* PyBobIpFacedetectFeatureExtractor_image(PyBobIpFacedetectFeatureExtractorObject* self, void*){
  BOB_TRY
  return PyBlitzArrayCxx_AsConstNumpy(self->cxx->getImage());
  BOB_CATCH_MEMBER("image could not be read", 0)
}

static auto model_indices = bob::extension::VariableDoc(
  "model_indices",
  "array_like <1D, int32>",
  "The indices at which the features are extracted, read and write access"
);
PyObject* PyBobIpFacedetectFeatureExtractor_get_model_indices(PyBobIpFacedetectFeatureExtractorObject* self, void*){
  BOB_TRY
  return PyBlitzArrayCxx_AsConstNumpy(self->cxx->getModelIndices());
  BOB_CATCH_MEMBER("model_indices could not be read", 0)
}
int PyBobIpFacedetectFeatureExtractor_set_model_indices(PyBobIpFacedetectFeatureExtractorObject* self, PyObject* value, void*){
  BOB_TRY
  PyBlitzArrayObject* data;
  if (!PyBlitzArray_Converter(value, &data)) return 0;
  auto data_ = make_safe(data);
  if (data->type_num != NPY_INT32 || data->ndim != 1){
    PyErr_Format(PyExc_TypeError, "model_indices can only be 1D and of type int32");
    return -1;
  }
  self->cxx->setModelIndices(*PyBlitzArrayCxx_AsBlitz<int32_t, 1>(data));
  return 0;
  BOB_CATCH_MEMBER("model_indices could not be set", -1)
}

static auto number_of_features = bob::extension::VariableDoc(
  "number_of_features",
  "int",
  "The length of the feature vector that will be extracted by this class, read access only"
);
PyObject* PyBobIpFacedetectFeatureExtractor_number_of_features(PyBobIpFacedetectFeatureExtractorObject* self, void*){
  BOB_TRY
  return Py_BuildValue("i", self->cxx->numberOfFeatures());
  BOB_CATCH_MEMBER("number_of_features could not be read", 0)
}

static auto number_of_labels = bob::extension::VariableDoc(
  "number_of_labels",
  "int",
  "The maximum label for the features in this class, read access only"
);
PyObject* PyBobIpFacedetectFeatureExtractor_number_of_labels(PyBobIpFacedetectFeatureExtractorObject* self, void*){
  BOB_TRY
  return Py_BuildValue("i", self->cxx->getMaxLabel());
  BOB_CATCH_MEMBER("number_of_labels could not be read", 0)
}

static auto extractors = bob::extension::VariableDoc(
  "extractors",
  "[:py:class:`bob.ip.base.LBP`]",
  "The LBP extractors, read access only"
);
PyObject* PyBobIpFacedetectFeatureExtractor_extractors(PyBobIpFacedetectFeatureExtractorObject* self, void*){
  BOB_TRY
  auto lbps = self->cxx->getExtractors();
  PyObject* list = PyList_New(lbps.size());
  if (!list) return 0;
  auto list_ = make_safe(list);
  for (Py_ssize_t i = 0; i < PyList_GET_SIZE(list); ++i){
    PyBobIpBaseLBPObject* lbp = reinterpret_cast<PyBobIpBaseLBPObject*>(PyBobIpBaseLBP_Type.tp_alloc(&PyBobIpBaseLBP_Type, 0));
    if (!lbp) return 0;
    lbp->cxx = lbps[i];
    PyList_SET_ITEM(list, i, Py_BuildValue("N", lbp));
  }
  return Py_BuildValue("O", list);
  BOB_CATCH_MEMBER("extractors could not be read", 0)
}

static auto patch_size = bob::extension::VariableDoc(
  "patch_size",
  "(int, int)",
  "The expected size of the patch that this extractor can handle, read access only"
);
PyObject* PyBobIpFacedetectFeatureExtractor_patch_size(PyBobIpFacedetectFeatureExtractorObject* self, void*){
  BOB_TRY
  return Py_BuildValue("ii", self->cxx->patchSize()[0], self->cxx->patchSize()[1]);
  BOB_CATCH_MEMBER("patch_size could not be read", 0)
}


static PyGetSetDef PyBobIpFacedetectFeatureExtractor_getseters[] = {
    {
      image.name(),
      (getter)PyBobIpFacedetectFeatureExtractor_image,
      0,
      image.doc(),
      0
    },
    {
      model_indices.name(),
      (getter)PyBobIpFacedetectFeatureExtractor_get_model_indices,
      (setter)PyBobIpFacedetectFeatureExtractor_set_model_indices,
      model_indices.doc(),
      0
    },
    {
      number_of_features.name(),
      (getter)PyBobIpFacedetectFeatureExtractor_number_of_features,
      0,
      number_of_features.doc(),
      0
    },
    {
      number_of_labels.name(),
      (getter)PyBobIpFacedetectFeatureExtractor_number_of_labels,
      0,
      number_of_labels.doc(),
      0
    },
    {
      extractors.name(),
      (getter)PyBobIpFacedetectFeatureExtractor_extractors,
      0,
      extractors.doc(),
      0
    },
    {
      patch_size.name(),
      (getter)PyBobIpFacedetectFeatureExtractor_patch_size,
      0,
      patch_size.doc(),
      0
    },
    {0}  /* Sentinel */
};

/******************************************************************/
/************ Functions Section ***********************************/
/******************************************************************/

static auto append = bob::extension::FunctionDoc(
  "append",
  "Appends the given feature extractor or LBP class to this one",
  "With this function you can either append a complete feature extractor, or a partial axtractor (i.e., a single LBP class) including the offset positions for them",
  true
)
.add_prototype("other")
.add_prototype("lbp, offsets")
.add_parameter("other", ":py:class:`FeatureExtractor`", "All LBP classes and offset positions of the given extractor will be appended")
.add_parameter("lbp", ":py:class:`bob.ip.base.LBP`", "The LBP extractor that will be added")
.add_parameter("offsets", "[(int,int)]", "The offset positions at which the given LBP will be extracted")
;

static PyObject* PyBobIpFacedetectFeatureExtractor_append(PyBobIpFacedetectFeatureExtractorObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY
  char** kwlist1 = append.kwlist(0);
  char** kwlist2 = append.kwlist(1);

  // get the number of command line arguments
  Py_ssize_t nargs = (args?PyTuple_Size(args):0) + (kwargs?PyDict_Size(kwargs):0);

  if (nargs == 1){
    PyBobIpFacedetectFeatureExtractorObject* other;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!", kwlist1, &PyBobIpFacedetectFeatureExtractor_Type, &other)) return 0;
    self->cxx->append(*other->cxx);
  } else {
    PyBobIpBaseLBPObject* lbp;
    PyObject* list;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O&O!", kwlist2, &PyBobIpBaseLBP_Converter, &lbp, &PyList_Type, &list)) return 0;
    auto lbp_ = make_safe(lbp);
    // extract list
    std::vector<blitz::TinyVector<int32_t, 2>> offsets(PyList_GET_SIZE(list));
    for (Py_ssize_t i = 0; i < PyList_GET_SIZE(list); ++i){
      if (!PyArg_ParseTuple(PyList_GET_ITEM(list,i), "hh", &offsets[i][0], &offsets[i][1])){
        PyErr_Format(PyExc_TypeError, "%s : expected a list of (int, int) tuples, but object number %d not", Py_TYPE(self)->tp_name, (int)i);
        return 0;
      }
    }
    self->cxx->append(lbp->cxx, offsets);
  }
  Py_RETURN_NONE;
  BOB_CATCH_MEMBER("cannot append", 0)
}

static auto extractor = bob::extension::FunctionDoc(
  "extractor",
  "Get the LBP feature extractor associated with the given feature index",
  0,
  true
)
.add_prototype("index", "lbp")
.add_parameter("index", "int", "The feature index for which the extractor should be retrieved")
.add_return("lbp", ":py:class:`bob.ip.base.LBP`", "The feature extractor for the given feature index")
;
static PyObject* PyBobIpFacedetectFeatureExtractor_extractor(PyBobIpFacedetectFeatureExtractorObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY
  char** kwlist = extractor.kwlist();

  int index;
  // by shape
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "i", kwlist, &index)) return 0;
  PyBobIpBaseLBPObject* lbp = reinterpret_cast<PyBobIpBaseLBPObject*>(PyBobIpBaseLBP_Type.tp_alloc(&PyBobIpBaseLBP_Type, 0));
  lbp->cxx = self->cxx->extractor(index);
  return Py_BuildValue("N", lbp);
  BOB_CATCH_MEMBER("cannot get extractor", 0)
}

static auto offset = bob::extension::FunctionDoc(
  "offset",
  "Get the offset position associated with the given feature index",
  0,
  true
)
.add_prototype("index", "offset")
.add_parameter("index", "int", "The feature index for which the extractor should be retrieved")
.add_return("offset", "(int,int)", "The offset position for the given feature index")
;
static PyObject* PyBobIpFacedetectFeatureExtractor_offset(PyBobIpFacedetectFeatureExtractorObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY
  char** kwlist = offset.kwlist();

  int index;
  // by shape
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "i", kwlist, &index)) return 0;
  auto offset = self->cxx->offset(index);
  return Py_BuildValue("ii", offset[0], offset[1]);
  BOB_CATCH_MEMBER("cannot get offset", 0)
}

static auto prepare = bob::extension::FunctionDoc(
  "prepare",
  "Take the given image to perform the next extraction steps for the given scale",
  "If ``compute_integral_square_image`` is enabled, the (internally stored) integral square image is computed as well. "
  "This image is required to compute the variance of the pixels in a given patch, see :py:func:`mean_variance`",
  true
)
.add_prototype("image, scale, [compute_integral_square_image]")
.add_parameter("image", "array_like <2D, uint8 or float>", "The image that should be used in the next extraction step")
.add_parameter("scale", "float", "The scale of the image to extract")
.add_parameter("compute_integral_square_image", "bool", "[Default: ``False``] : Enable the computation of the integral square image")
;
static PyObject* PyBobIpFacedetectFeatureExtractor_prepare(PyBobIpFacedetectFeatureExtractorObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY
  char** kwlist = prepare.kwlist();

  PyBlitzArrayObject* image;
  double scale;
  PyObject* cisi = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O&d|O!", kwlist, &PyBlitzArray_Converter, &image, &scale, &PyBool_Type, &cisi)){
    return 0;
  }
  auto image_ = make_safe(image);
  if (image->ndim != 2){
    PyErr_Format(PyExc_TypeError, "%s : The input image must be 2D, not %dD", Py_TYPE(self)->tp_name, (int)image->ndim);
    return 0;
  }
  switch (image->type_num){
    case NPY_UINT8: self->cxx->prepare(*PyBlitzArrayCxx_AsBlitz<uint8_t,2>(image), scale, f(cisi)); Py_RETURN_NONE;
    case NPY_FLOAT64: self->cxx->prepare(*PyBlitzArrayCxx_AsBlitz<double,2>(image), scale, f(cisi)); Py_RETURN_NONE;
    default:
      PyErr_Format(PyExc_TypeError, "%s : The input image must be of type uint8 or float", Py_TYPE(self)->tp_name);
      return 0;
  }
  BOB_CATCH_MEMBER("cannot prepare image", 0)
}

static auto extract_all = bob::extension::FunctionDoc(
  "extract_all",
  "Extracts all features into the given dataset of (training) features at the given index",
  "This function exists to extract training features for several training patches. "
  "To avoid data copying, the full training dataset, and the current training feature index need to be provided.",
  true
)
.add_prototype("bounding_box, dataset, dataset_index")
.add_parameter("bounding_box", ":py:class:`BoundingBox`", "The bounding box for which the features should be extracted")
.add_parameter("dataset", "array_like <2D, uint16>", "The (training) dataset, into which the features should be extracted; must be of shape (#training_patches, :py:attr:`number_of_features`)")
.add_parameter("dataset_index", "int", "The index of the current training patch")
;
static PyObject* PyBobIpFacedetectFeatureExtractor_extract_all(PyBobIpFacedetectFeatureExtractorObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY
  char** kwlist = extract_all.kwlist();

  PyBobIpFacedetectBoundingBoxObject* bb;
  PyBlitzArrayObject* dataset;
  int index;
  // by shape
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!O&i", kwlist, &PyBobIpFacedetectBoundingBox_Type, &bb, &PyBlitzArray_OutputConverter, &dataset, &index)){
    return 0;
  }
  auto dataset_ = make_safe(dataset);
  auto ds = PyBlitzArrayCxx_AsBlitz<uint16_t, 2>(dataset, "dataset");
  if (!ds) return 0;
  self->cxx->extractAll(*bb->cxx, *ds, index);
  Py_RETURN_NONE;
  BOB_CATCH_MEMBER("cannot extract all features", 0)
}

static auto extract_indexed = bob::extension::FunctionDoc(
  "extract_indexed",
  "Extracts the features only at the required locations, which defaults to :py:attr:`model_indices`",
  0,
  true
)
.add_prototype("bounding_box, feature_vector, [indices]")
.add_parameter("bounding_box", ":py:class:`BoundingBox`", "The bounding box for which the features should be extracted")
.add_parameter("feature_vector", "array_like <1D, uint16>", "The feature vector, into which the features should be extracted; must be of size :py:attr:`number_of_features`")
.add_parameter("indices", "array_like<1D,int32>", "The indices, for which the features should be extracted; if not given, :py:attr:`model_indices` is used (must be set beforehands)")
;
static PyObject* PyBobIpFacedetectFeatureExtractor_extract_indexed(PyBobIpFacedetectFeatureExtractorObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY
  char** kwlist = extract_indexed.kwlist();

  PyBobIpFacedetectBoundingBoxObject* bb;
  PyBlitzArrayObject* fv, *indices = 0;
  // by shape
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!O&|O&", kwlist, &PyBobIpFacedetectBoundingBox_Type, &bb, &PyBlitzArray_OutputConverter, &fv, &PyBlitzArray_Converter, &indices)){
    return 0;
  }
  auto fv_ = make_safe(fv);
  auto indices_ = make_xsafe(indices);
  auto f = PyBlitzArrayCxx_AsBlitz<uint16_t, 1>(fv, "feature_vector");
  if (!f) return 0;

  if (indices){
    auto i = PyBlitzArrayCxx_AsBlitz<int32_t, 1>(indices, "indices");
    if (!i) return 0;
    self->cxx->extractIndexed(*bb->cxx, *f, *i);
  } else {
    self->cxx->extractSome(*bb->cxx, *f);
  }
  Py_RETURN_NONE;
  BOB_CATCH_MEMBER("cannot extract indexed features", 0)
}

static auto mean_variance = bob::extension::FunctionDoc(
  "mean_variance",
  "Computes the mean (and the variance) of the pixel gray values in the given bounding box",
  0,
  true
)
.add_prototype("bounding_box, [compute_variance]", "mv")
.add_parameter("bounding_box", ":py:class:`BoundingBox`", "The bounding box for which the mean (and variance) shoulf be calculated")
.add_parameter("compute_variance", "bool", "[Default: ``False``] If enabled, the variance is computed as well; requires the ``compute_integral_square_image`` enabled in the :py:func:`prepare` function")
.add_return("mv", "float or (float, float)", "The mean (or the mean and the variance) of the pixel gray values for the given bounding box")
;
static PyObject* PyBobIpFacedetectFeatureExtractor_mean_variance(PyBobIpFacedetectFeatureExtractorObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY
  char** kwlist = mean_variance.kwlist();

  PyBobIpFacedetectBoundingBoxObject* bb;
  PyObject* cv = 0;
  // by shape
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!|O!", kwlist, &PyBobIpFacedetectBoundingBox_Type, &bb, &PyBool_Type, &cv)){
    return 0;
  }

  if (f(cv)){
    auto res = self->cxx->meanAndVariance(*bb->cxx);
    return Py_BuildValue("dd", res[0], res[1]);
  } else {
    double res = self->cxx->mean(*bb->cxx);
    return Py_BuildValue("d", res);
  }
  BOB_CATCH_MEMBER("cannot compute mean (and variance)", 0)
}

static auto load = bob::extension::FunctionDoc(
  "load",
  "Loads the extractors from the given HDF5 file",
  0,
  true
)
.add_prototype("hdf5")
.add_parameter("hdf5", ":py:class:`bob.io.base.HDF5File`", "The file to read from")
;
static PyObject* PyBobIpFacedetectFeatureExtractor_load(PyBobIpFacedetectFeatureExtractorObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY
  char** kwlist = load.kwlist();

  PyBobIoHDF5FileObject* hdf5;
  // by shape
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O&", kwlist, &PyBobIoHDF5File_Converter, &hdf5)) return 0;

  self->cxx->load(*hdf5->f);
  Py_RETURN_NONE;
  BOB_CATCH_MEMBER("cannot load", 0)
}

static auto save = bob::extension::FunctionDoc(
  "save",
  "Saves the extractors to the given HDF5 file",
  0,
  true
)
.add_prototype("hdf5")
.add_parameter("hdf5", ":py:class:`bob.io.base.HDF5File`", "The file to write to")
;
static PyObject* PyBobIpFacedetectFeatureExtractor_save(PyBobIpFacedetectFeatureExtractorObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY
  char** kwlist = save.kwlist();

  PyBobIoHDF5FileObject* hdf5;
  // by shape
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O&", kwlist, &PyBobIoHDF5File_Converter, &hdf5)) return 0;

  auto hdf5_ = make_safe(hdf5);

  self->cxx->save(*hdf5->f);
  Py_RETURN_NONE;
  BOB_CATCH_MEMBER("cannot save", 0)
}


static PyMethodDef PyBobIpFacedetectFeatureExtractor_methods[] = {
  {
    append.name(),
    (PyCFunction)PyBobIpFacedetectFeatureExtractor_append,
    METH_VARARGS|METH_KEYWORDS,
    append.doc()
  },
  {
    extractor.name(),
    (PyCFunction)PyBobIpFacedetectFeatureExtractor_extractor,
    METH_VARARGS|METH_KEYWORDS,
    extractor.doc()
  },
  {
    offset.name(),
    (PyCFunction)PyBobIpFacedetectFeatureExtractor_offset,
    METH_VARARGS|METH_KEYWORDS,
    offset.doc()
  },
  {
    prepare.name(),
    (PyCFunction)PyBobIpFacedetectFeatureExtractor_prepare,
    METH_VARARGS|METH_KEYWORDS,
    prepare.doc()
  },
  {
    extract_all.name(),
    (PyCFunction)PyBobIpFacedetectFeatureExtractor_extract_all,
    METH_VARARGS|METH_KEYWORDS,
    extract_all.doc()
  },
  {
    extract_indexed.name(),
    (PyCFunction)PyBobIpFacedetectFeatureExtractor_extract_indexed,
    METH_VARARGS|METH_KEYWORDS,
    extract_indexed.doc()
  },
  {
    mean_variance.name(),
    (PyCFunction)PyBobIpFacedetectFeatureExtractor_mean_variance,
    METH_VARARGS|METH_KEYWORDS,
    mean_variance.doc()
  },
  {
    load.name(),
    (PyCFunction)PyBobIpFacedetectFeatureExtractor_load,
    METH_VARARGS|METH_KEYWORDS,
    load.doc()
  },
  {
    save.name(),
    (PyCFunction)PyBobIpFacedetectFeatureExtractor_save,
    METH_VARARGS|METH_KEYWORDS,
    save.doc()
  },
  {0} /* Sentinel */
};

/******************************************************************/
/************ Module Section **************************************/
/******************************************************************/

// Define the DCTFeatures type struct; will be initialized later
PyTypeObject PyBobIpFacedetectFeatureExtractor_Type = {
  PyVarObject_HEAD_INIT(0,0)
  0
};

bool init_BobIpFacedetectFeatureExtractor(PyObject* module)
{
  // initialize the type struct
  PyBobIpFacedetectFeatureExtractor_Type.tp_name = FeatureExtractor_doc.name();
  PyBobIpFacedetectFeatureExtractor_Type.tp_basicsize = sizeof(PyBobIpFacedetectFeatureExtractorObject);
  PyBobIpFacedetectFeatureExtractor_Type.tp_flags = Py_TPFLAGS_DEFAULT;
  PyBobIpFacedetectFeatureExtractor_Type.tp_doc = FeatureExtractor_doc.doc();
//  PyBobIpFacedetectFeatureExtractor_Type.tp_repr = (reprfunc)PyBobIpFacedetectFeatureExtractor_Str;
//  PyBobIpFacedetectFeatureExtractor_Type.tp_str = (reprfunc)PyBobIpFacedetectFeatureExtractor_Str;

  // set the functions
  PyBobIpFacedetectFeatureExtractor_Type.tp_new = PyType_GenericNew;
  PyBobIpFacedetectFeatureExtractor_Type.tp_init = reinterpret_cast<initproc>(PyBobIpFacedetectFeatureExtractor_init);
  PyBobIpFacedetectFeatureExtractor_Type.tp_dealloc = reinterpret_cast<destructor>(PyBobIpFacedetectFeatureExtractor_delete);
//  PyBobIpFacedetectFeatureExtractor_Type.tp_richcompare = reinterpret_cast<richcmpfunc>(PyBobIpFacedetectFeatureExtractor_RichCompare);
  PyBobIpFacedetectFeatureExtractor_Type.tp_methods = PyBobIpFacedetectFeatureExtractor_methods;
  PyBobIpFacedetectFeatureExtractor_Type.tp_getset = PyBobIpFacedetectFeatureExtractor_getseters;

  // check that everything is fine
  if (PyType_Ready(&PyBobIpFacedetectFeatureExtractor_Type) < 0)
    return false;

  // add the type to the module
  Py_INCREF(&PyBobIpFacedetectFeatureExtractor_Type);
  return PyModule_AddObject(module, "FeatureExtractor", (PyObject*)&PyBobIpFacedetectFeatureExtractor_Type) >= 0;
}
