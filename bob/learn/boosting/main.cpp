#ifdef NO_IMPORT_ARRAY
#undef NO_IMPORT_ARRAY
#endif // NO_IMPORT_ARRAY

#include "main.h"
#include <bob.learn.boosting/Functions.h>

auto weighted_histogram_doc = bob::extension::FunctionDoc(
  "weighted_histogram",
  "Computes a weighted histogram from the given features."
)
.add_prototype("features, weights, histogram")
.add_parameter("features", "array_like <1D, uint16>", "The vector of features to compute a histogram for")
.add_parameter("weights", "array_like <1D, float>", "The vector of weights; must be of the same size as the features")
.add_parameter("histogram", "array_like <1D, float>", "The histogram that will be filled")
;

PyObject* weighted_histogram(PyObject* args, PyObject* kwargs){
  char* kwlist[] = {c("features"), c("weights"), c("histogram"), NULL};

  PyBlitzArrayObject* features,* weights,* histogram;
  if (!PyArg_ParseTupleAndKeywords(
    args, kwargs,
    "O&O&O&", kwlist, &PyBlitzArray_Converter, &features, &PyBlitzArray_Converter, &weights, &PyBlitzArray_OutputConverter, &histogram
  )){
    return NULL;
  }

  auto _1 = make_safe(features), _2 = make_safe(weights), _3 = make_safe(histogram);

  // tests
  if (features->type_num != NPY_UINT16 || features->ndim != 1){
    PyErr_Format(PyExc_RuntimeError, "weighted_histogram: features parameter must be 1D of numpy.uint16");
    return NULL;
  }
  if (weights->type_num != NPY_FLOAT16 || weights->ndim != 1){
    PyErr_Format(PyExc_RuntimeError, "weighted_histogram: weights parameter must be 1D of numpy.float64");
    return NULL;
  }
  if (histogram->type_num != NPY_FLOAT16 || histogram->ndim != 1){
    PyErr_Format(PyExc_RuntimeError, "weighted_histogram: histogram parameter must be 1D of numpy.float64");
    return NULL;
  }
  bob::learn::boosting::weighted_histogram(
    *PyBlitzArrayCxx_AsBlitz<uint16_t,1>(features),
    *PyBlitzArrayCxx_AsBlitz<double,1>(weights),
    *PyBlitzArrayCxx_AsBlitz<double,1>(histogram)
  );

  Py_RETURN_NONE;

}

static PyMethodDef BoostingMethods[] = {
  {
    weighted_histogram_doc.name(),
    &weighted_histogram,
    METH_VARARGS | METH_KEYWORDS,
    weighted_histogram_doc.doc()
  },
  {NULL}
};


static const char* const module_docstr = "C++ implementations for several classes and functions in the bob.boosting module";

#if PY_VERSION_HEX >= 0x03000000
static PyModuleDef module_definition = {
  PyModuleDef_HEAD_INIT,
  BOB_EXT_MODULE_NAME,
  module_docstr,
  -1,
  BoostingMethods,
  0,
};
#endif


PyObject* create_module(void)
{

# if PY_VERSION_HEX >= 0x03000000
  PyObject* module = PyModule_Create(&module_definition);
  auto module_ = make_xsafe(module);
  const char* ret = "O";
# else
  PyObject* module = Py_InitModule3(BOB_EXT_MODULE_NAME, BoostingMethods, module_docstr);
  const char* ret = "N";
# endif
  if (!module) return 0;

  if (!init_LossFunction(module)) return NULL;
  if (!init_JesorskyLoss(module)) return NULL;


  if (!init_WeakMachine(module)) return NULL;
  if (!init_StumpMachine(module)) return NULL;
  if (!init_LUTMachine(module)) return NULL;
  if (!init_BoostedMachine(module)) return NULL;

  if (!init_LUTTrainer(module)) return NULL;


  /* imports C-API dependencies */
  if (import_bob_blitz() < 0) return NULL;
  if (import_bob_core_logging() < 0) return NULL;
  if (import_bob_io_base() < 0) return NULL;

  // module was initialized successfully
  return Py_BuildValue(ret, module);
}

PyMODINIT_FUNC BOB_EXT_ENTRY_NAME (void) {
# if PY_VERSION_HEX >= 0x03000000
  return
# endif
    create_module();
}
