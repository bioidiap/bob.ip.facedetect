
#include "main.h"
#include <boost/type_index.hpp>
#include <boost/functional/hash.hpp>

static auto stumpMachine_doc = bob::extension::ClassDoc(
  "StumpMachine",
  "A weak machine that bases it's decision on comparing the given value to a threshold",
  ".. todo:: Improve documentation."
)
.add_constructor(
  bob::extension::FunctionDoc(
    "__init__",
    "Initializes a StumpMachine object.",
    0,
    true
  )
  .add_prototype("threshold, polarity, index", "")
  .add_prototype("hdf5", "")
  .add_parameter("threshold", "float", "The decision threshold")
  .add_parameter("polarity", "float", "-1 if positive values are below threshold, +1 if positive values are above threshold")
  .add_parameter("index", "int", "The index into the feature vector that is thresholded")
  .add_parameter("hdf5", ":py:class:`bob.io.base.HDF5File`", "The HDF5 file object to read the weak classifier from")
);


// Some functions
static int stumpMachine_init(
  StumpMachineObject* self,
  PyObject* args,
  PyObject* kwargs
)
{
  Py_ssize_t argument_count = (args ? PyTuple_Size(args) : 0) + (kwargs ? PyDict_Size(kwargs) : 0);

  try{
    switch (argument_count){
      case 1:{
        char*  kwlist[] = {c("hdf5"), NULL};
        PyBobIoHDF5FileObject* file = 0;
        if (
          PyArg_ParseTupleAndKeywords(args, kwargs,
              "O&", kwlist,
              PyBobIoHDF5File_Converter, &file
          )
        ){
          // construct from HDF5File
          auto _ = make_safe(file);
          self->base.reset(new bob::learn::boosting::StumpMachine(*file->f));
        } else {
          stumpMachine_doc.print_usage();
          return -1;
        }
      } break;

      case 3:{
        char* kwlist[] = {c("threshold"), c("polarity"), c("index"), NULL};
        double threshold, polarity;
        int index;
        if (
          PyArg_ParseTupleAndKeywords(args, kwargs,
              "ddi", kwlist,
              &threshold, &polarity, &index
          )
        ){
          // construct with parameters
          self->base.reset(new bob::learn::boosting::StumpMachine(threshold, polarity, index));
        } else {
          stumpMachine_doc.print_usage();
          return -1;
        }
      } break;
      default:
        stumpMachine_doc.print_usage();
        PyErr_Format(PyExc_RuntimeError, "number of arguments mismatch - %s requires 1 or 3 arguments, but you provided %" PY_FORMAT_SIZE_T "d", Py_TYPE(self)->tp_name, argument_count);
        return -1;
    }
  } catch (std::exception& ex) {
    PyErr_SetString(PyExc_RuntimeError, ex.what());
    return -1;
  }
  catch (...) {
    PyErr_Format(PyExc_RuntimeError, "cannot create new object of type `%s' - unknown exception thrown", Py_TYPE(self)->tp_name);
    return -1;
  }

  // set the base class as well
  self->parent.base = self->base;

  return 0;
}

static void stumpMachine_exit(
  StumpMachineObject* self
)
{
  self->base.reset();
  self->parent.base.reset();
  Py_TYPE(self)->tp_free((PyObject*)self);
}



static auto stumpMachine_threshold_doc = bob::extension::VariableDoc(
  "threshold",
  "float",
  "The thresholds that the feature value will be compared with"
);

static PyObject* stumpMachine_threshold(
  StumpMachineObject* self,
  void* )
{
  return Py_BuildValue("d", self->base->getThreshold());
}

static auto stumpMachine_polarity_doc = bob::extension::VariableDoc(
  "polarity",
  "float",
  "The polarity of the comparison -1 if the values lower than the threshold should be accepted, +1 otherwise."
);
static PyObject* stumpMachine_polarity(
  StumpMachineObject* self,
  void* )
{
  return Py_BuildValue("d", self->base->getPolarity());
}



static auto stumpMachine_forward_doc = bob::extension::FunctionDoc(
  "forward",
  "Returns the prediction for the given feature vector(s)",
  ".. note:: The ``__call__`` function is an alias for this function.\n\n"
  ".. todo:: write more detailed documentation",
  true
)
.add_prototype("features", "prediction")
.add_prototype("features, predictions")
.add_parameter("features", "float <#inputs> or float <#samples, #inputs>", "The feature vector(s) the prediction should be computed for. If only a single feature is given, the resulting prediction is returned as a float. Otherwise it is stored in the second ``predictions`` parameter.")
.add_parameter("predictions", "float <#samples> or float <#samples, 1>", "The predicted values -- in case several ``features`` are provided.")
.add_return("prediction", "float", "The predicted value -- in case a single feature is provided")
;

template <typename T, int N> void _forward(StumpMachineObject* self, PyBlitzArrayObject* features, PyBlitzArrayObject* predictions){
  const auto f = PyBlitzArrayCxx_AsBlitz<T,2>(features);
  auto p = PyBlitzArrayCxx_AsBlitz<double,N>(predictions);
  self->base->forward(*f, *p);
}

static PyObject* stumpMachine_forward(
  StumpMachineObject* self,
  PyObject* args,
  PyObject* kwargs
)
{
  // get list of arguments
  char* kwlist[] = {c("features"), c("predictions"), NULL};

  PyBlitzArrayObject* p_features = 0,* p_predictions = 0;

  if (!PyArg_ParseTupleAndKeywords(
          args, kwargs,
          "O&|O&", kwlist,
          &PyBlitzArray_Converter, &p_features,
          &PyBlitzArray_Converter, &p_predictions)
  )
    return NULL;

  auto _1 = make_safe(p_features), _2 = make_xsafe(p_predictions);

  try{
    const char* n1 = PyBlitzArray_TypenumAsString(NPY_UINT16);
    const char* n2 = PyBlitzArray_TypenumAsString(NPY_FLOAT64);
    // check for the different ways, the function can be called
    if (p_features->type_num != NPY_UINT16 && p_features->type_num != NPY_FLOAT64){
      PyErr_Format(PyExc_TypeError, "The parameter 'features' only supports 1D or 2D arrays of types '%s' or '%s'", n1, n2);
      return NULL;
    }
    if (p_features->ndim == 1 && !p_predictions){
      // first way
      double prediction;
      switch (p_features->type_num){
        case NPY_UINT16:{
          const auto inputs = PyBlitzArrayCxx_AsBlitz<uint16_t,1>(p_features);
          prediction = self->base->forward(*inputs);
          break;
        }
        case NPY_FLOAT64:{
          const auto inputs = PyBlitzArrayCxx_AsBlitz<double,1>(p_features);
          prediction = self->base->forward(*inputs);
          break;
        }
        default:
          // already handled
          return NULL;
      }
      return Py_BuildValue("d", prediction);
    } else if (p_features->ndim == 2 && p_predictions){
      if (p_predictions->type_num != NPY_FLOAT64){
        PyErr_Format(PyExc_TypeError, "The parameter 'predictions' only supports 1D or 2D arrays of type '%s'", n2);
        return NULL;
      }
      switch (p_predictions->ndim){
        case 1:
          switch (p_features->type_num){
            case NPY_UINT16: _forward<uint16_t,1>(self, p_features, p_predictions); break;
            case NPY_FLOAT64: _forward<double,1>(self, p_features, p_predictions); break;
            default: return NULL;
          }
          break;
        case 2:
          switch (p_features->type_num){
            case NPY_UINT16: _forward<uint16_t,2>(self, p_features, p_predictions); break;
            case NPY_FLOAT64: _forward<double,2>(self, p_features, p_predictions); break;
            default: return NULL;
          }
          break;
        default:
          PyErr_Format(PyExc_TypeError, "The parameter 'predictions' only supports 1D or 2D arrays of type '%s'", n2);
          return NULL;
      }
      Py_RETURN_NONE;
    } else {
      PyErr_BadArgument();
      return NULL;
    }
  } catch (std::exception& ex) {
    PyErr_SetString(PyExc_RuntimeError, ex.what());
    return NULL;
  }
  catch (...) {
    PyErr_Format(PyExc_RuntimeError, "cannot create new object of type `%s' - unknown exception thrown", Py_TYPE(self)->tp_name);
    return NULL;
  }
}


static auto stumpMachine_getIndices_doc = bob::extension::FunctionDoc(
  "feature_indices",
  "Returns the feature index that will be used in this weak machine",
  NULL,
  true
)
.add_prototype("", "indices")
.add_return("indices", "int32 <1>", "The feature index required by this machine")
;

static PyObject* stumpMachine_getIndices(
  StumpMachineObject* self,
  PyObject* args,
  PyObject* kwargs
)
{
  // get list of arguments
  char* kwlist[] = {NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "", kwlist))
    return NULL;

  auto retval = self->base->getIndices();
  return PyBlitzArrayCxx_AsConstNumpy(retval);
}


static auto stumpMachine_load_doc = bob::extension::FunctionDoc(
  "load",
  "Loads the Stump machine from the given HDF5 file",
  NULL,
  true
)
.add_prototype("hdf5")
.add_parameter("hdf5", ":py:class:`bob.io.base.HDF5File`", "The HDF5 file to load this weak machine from.")
;

static PyObject* stumpMachine_load(
  StumpMachineObject* self,
  PyObject* args,
  PyObject* kwargs
)
{
  // get list of arguments
  char* kwlist[] = {c("hdf5"), NULL};
  PyBobIoHDF5FileObject* file = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs,
        "O&", kwlist,
        PyBobIoHDF5File_Converter, &file
    )
  ){
    stumpMachine_load_doc.print_usage();
    return NULL;
  }

  auto _1 = make_safe(file);
  self->base->load(*file->f);
  Py_RETURN_NONE;
}


static auto stumpMachine_save_doc = bob::extension::FunctionDoc(
  "save",
  "Saves the content of this machine to the given HDF5 file",
  NULL,
  true
)
.add_prototype("hdf5")
.add_parameter("hdf5", ":py:class:`bob.io.base.HDF5File`", "The HDF5 file to save this weak machine to.")
;

static PyObject* stumpMachine_save(
  StumpMachineObject* self,
  PyObject* args,
  PyObject* kwargs
)
{
  // get list of arguments
  char* kwlist[] = {c("hdf5"), NULL};
  PyBobIoHDF5FileObject* file = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs,
        "O&", kwlist,
        PyBobIoHDF5File_Converter, &file
    )
  ){
    stumpMachine_save_doc.print_usage();
    return NULL;
  }

  auto _1 = make_safe(file);
  self->base->save(*file->f);
  Py_RETURN_NONE;
}

// bind the class
static PyGetSetDef stumpMachine_Getters[] = {
  {
    stumpMachine_threshold_doc.name(),
    (getter)stumpMachine_threshold,
    NULL,
    stumpMachine_threshold_doc.doc(),
    NULL
  },
  {
    stumpMachine_polarity_doc.name(),
    (getter)stumpMachine_polarity,
    NULL,
    stumpMachine_polarity_doc.doc(),
    NULL
  },
  {NULL}
};

static PyMethodDef stumpMachine_Methods[] = {
  {
    stumpMachine_forward_doc.name(),
    (PyCFunction)stumpMachine_forward,
    METH_VARARGS | METH_KEYWORDS,
    stumpMachine_forward_doc.doc(),
  },
  {
    stumpMachine_getIndices_doc.name(),
    (PyCFunction)stumpMachine_getIndices,
    METH_VARARGS | METH_KEYWORDS,
    stumpMachine_getIndices_doc.doc(),
  },
  {
    stumpMachine_load_doc.name(),
    (PyCFunction)stumpMachine_load,
    METH_VARARGS | METH_KEYWORDS,
    stumpMachine_load_doc.doc(),
  },
  {
    stumpMachine_save_doc.name(),
    (PyCFunction)stumpMachine_save,
    METH_VARARGS | METH_KEYWORDS,
    stumpMachine_save_doc.doc(),
  },
  {NULL}
};


// Define Jesorsky Loss Type object; will be filled later
PyTypeObject StumpMachineType = {
  PyVarObject_HEAD_INIT(0,0)
  0
};


PyObject* stumpMachineCreate(boost::shared_ptr<bob::learn::boosting::WeakMachine> machine){
  PyObject* o = StumpMachineType.tp_alloc(&StumpMachineType,0);
  reinterpret_cast<StumpMachineObject*>(o)->base = boost::dynamic_pointer_cast<bob::learn::boosting::StumpMachine>(machine);
  reinterpret_cast<StumpMachineObject*>(o)->parent.base = machine;
  return o;
}


bool init_StumpMachine(PyObject* module)
{

  // initialize the JesorskyLossType struct
  StumpMachineType.tp_name = stumpMachine_doc.name();
  StumpMachineType.tp_basicsize = sizeof(StumpMachineObject);
  StumpMachineType.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
  StumpMachineType.tp_doc = stumpMachine_doc.doc();
  StumpMachineType.tp_base = &WeakMachineType;

  // set the functions
  StumpMachineType.tp_new = PyType_GenericNew;
  StumpMachineType.tp_init = reinterpret_cast<initproc>(stumpMachine_init);
  StumpMachineType.tp_dealloc = reinterpret_cast<destructor>(stumpMachine_exit);
  StumpMachineType.tp_call = reinterpret_cast<ternaryfunc>(stumpMachine_forward);
  StumpMachineType.tp_getset = stumpMachine_Getters;
  StumpMachineType.tp_methods = stumpMachine_Methods;

  // register machine
  if (!registerMachineType(boost::typeindex::type_id<bob::learn::boosting::StumpMachine>().hash_code(), &stumpMachineCreate))
    return false;

  // check that everyting is fine
  if (PyType_Ready(&StumpMachineType) < 0)
    return false;

  // add the type to the module
  Py_INCREF(&StumpMachineType);
  return PyModule_AddObject(module, stumpMachine_doc.name(), (PyObject*)&StumpMachineType) >= 0;
}

