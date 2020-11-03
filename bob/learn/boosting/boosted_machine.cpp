#include "main.h"

static auto boostedMachine_doc = bob::extension::ClassDoc(
  "BoostedMachine",
  "A strong machine that holds a weighted combination of weak machines",
  ".. todo:: Improve documentation."
)
.add_constructor(
  bob::extension::FunctionDoc(
    "__init__",
    "Initializes a BoostedMachine object",
    "",
    true
  )
  .add_prototype("", "")
//  .add_prototype("weak_classifiers, weights", "")
  .add_prototype("hdf5", "")
//  .add_parameter("weak_classifiers", "[bob.boosting.machine.WeakMachine]", "A list of weak machines that should be used in this strong machine")
//  .add_parameter("weights", "float <#machines,#outputs>", "The list of weights for the machines.")
  .add_parameter("hdf5", ":py:class:`bob.io.base.HDF5File`", "The HDF5 file object to read the weak classifier from")
);


// Some functions
static int boostedMachine_init(
  BoostedMachineObject* self,
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
          self->base.reset(new bob::learn::boosting::BoostedMachine(*file->f));
          return 0;
        }
        boostedMachine_doc.print_usage();
        return -1;
      } break;

      case 0:{
        self->base.reset(new bob::learn::boosting::BoostedMachine());
        return 0;
      }
      default:
        boostedMachine_doc.print_usage();
        PyErr_Format(PyExc_RuntimeError, "number of arguments mismatch - %s requires 0 or 1 arguments, but you provided %" PY_FORMAT_SIZE_T "d", Py_TYPE(self)->tp_name, argument_count);
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

  return 0;
}

static void boostedMachine_exit(
  BoostedMachineObject* self
)
{
  self->base.reset();
  Py_TYPE(self)->tp_free(reinterpret_cast<PyObject*>(self));
}



static auto boostedMachine_indices_doc = bob::extension::VariableDoc(
  "indices",
  "int <#machines,#outputs>",
  "The indices into the feature vector required by all of the weak machines."
);

static PyObject* boostedMachine_indices(
  BoostedMachineObject* self,
  void*
)
{
  auto retval = self->base->getIndices();
  return PyBlitzArrayCxx_AsConstNumpy(retval);
}


static auto boostedMachine_weights_doc = bob::extension::VariableDoc(
  "weights",
  "float <#machines,#outputs>",
  "The weights for the weak machines"
);

static PyObject* boostedMachine_weights(
  BoostedMachineObject* self,
  void*
)
{
  auto retval = self->base->getWeights();
  return PyBlitzArrayCxx_AsConstNumpy(retval);
}


static auto boostedMachine_outputs_doc = bob::extension::VariableDoc(
  "outputs",
  "int",
  "The number of outputs; for uni-variate classifiers always 1"
);

static PyObject* boostedMachine_outputs(
  BoostedMachineObject* self,
  void*
)
{
  return Py_BuildValue("i", self->base->numberOfOutputs());
}


static auto boostedMachine_machines_doc = bob::extension::VariableDoc(
  "weak_machines",
  "[:py:class:`WeakMachine`]",
  "The list of weak machines stored in this strong machine"
);

static PyObject* boostedMachine_machines(
  BoostedMachineObject* self,
  void*
)
{
  // create new list
  auto machines = self->base->getWeakMachines();
  PyObject* list = PyList_New(machines.size());

  // fill list
  for (unsigned i = 0; i < machines.size(); ++i){
    PyObject* machine = createMachine(machines[i]);
    if (!machine) return NULL;
    PyList_SetItem(list, i, machine);
  }

  return reinterpret_cast<PyObject*>(list);
}



static auto boostedMachine_add_doc = bob::extension::FunctionDoc(
  "add_weak_machine",
  "Adds the given weak machine and its weight(s) to the list of weak machines",
  NULL,
  true
)
.add_prototype("machine, weight")
.add_prototype("machine, weights")
.add_parameter("machine", "A derivative from :py:class:`WeakMachine`", "The weak machine to add")
.add_parameter("weight", "float", "The weight for the machine (uni-variate)")
.add_parameter("weights", "float <#outputs>", "The weights for the machine (multi-variate)")
;

static PyObject* boostedMachine_add(
  BoostedMachineObject* self,
  PyObject* args,
  PyObject* kwargs
)
{
  try{
    Py_ssize_t argument_count = (args ? PyTuple_Size(args) : 0) + (kwargs ? PyDict_Size(kwargs) : 0);

    if (argument_count != 2){
      boostedMachine_add_doc.print_usage();
      PyErr_Format(PyExc_RuntimeError, "number of arguments mismatch - %s requires 2 arguments, but you provided %" PY_FORMAT_SIZE_T "d", Py_TYPE(self)->tp_name, argument_count);
    }

    // two ways of calling the function (two different kwargs)
    char* kwlist1[] = {c("machine"), c("weight"), NULL};
    char* kwlist2[] = {c("machine"), c("weights"), NULL};

    PyObject* key = Py_BuildValue("s", kwlist1[1]);
    auto _k = make_safe(key);
    if (
      (kwargs && (PyDict_Contains(kwargs, key))) ||
      (args && PyFloat_Check(PyTuple_GetItem(args, PyTuple_Size(args)-1)))
    ){
      WeakMachineObject* p_machine = 0;
      double weight;
      // single weight
      if (!PyArg_ParseTupleAndKeywords(
          args, kwargs,
          "O&d", kwlist1,
          &weakMachineConverter, &p_machine,
          &weight
        )
      ){
        boostedMachine_add_doc.print_usage();
        return NULL;
      }
      auto _1 = make_safe(p_machine);
      self->base->add_weak_machine(p_machine->base, weight);
    } else {
      WeakMachineObject* p_machine = 0;
      PyBlitzArrayObject* p_weights = 0;
      // single weight
      if (!PyArg_ParseTupleAndKeywords(
          args, kwargs,
          "O&O&", kwlist2,
          &weakMachineConverter, &p_machine,
          &PyBlitzArray_Converter, &p_weights
        )
      ){
        boostedMachine_add_doc.print_usage();
        return NULL;
      }
      auto _1 = make_safe(p_machine);
      auto _2 = make_safe(p_weights);
      const auto weights = PyBlitzArrayCxx_AsBlitz<double,1>(p_weights, kwlist2[1]);
      if (!weights){
        boostedMachine_add_doc.print_usage();
        return NULL;
      }
      self->base->add_weak_machine(p_machine->base, *weights);
    }
  } catch (std::exception& ex) {
    PyErr_SetString(PyExc_RuntimeError, ex.what());
    return NULL;
  }
  catch (...) {
    PyErr_Format(PyExc_RuntimeError, "cannot create new object of type `%s' - unknown exception thrown", Py_TYPE(self)->tp_name);
    return NULL;
  }
  Py_RETURN_NONE;
}


static auto boostedMachine_forward_doc = bob::extension::FunctionDoc(
  "forward",
  "Returns the prediction for the given feature vector(s)",
  ".. note:: The ``__call__`` function is an alias for this function.\n\n"
  "This function can be called in six different ways:\n\n"
  "1. ``(uint16 <#inputs>)`` will compute and return the uni-variate prediction for a single feature vector.\n"
  "2. ``(uint16 <#samples,#inputs>, float <#samples>)`` will compute the uni-variate prediction for several feature vectors.\n"
  "3. ``(uint16 <#samples,#inputs>, float <#samples>, float<#samples>)`` will compute the uni-variate prediction and the labels for several feature vectors.\n"
  "4. ``(uint16 <#inputs>, float <#outputs>)`` will compute the multi-variate prediction for a single feature vector.\n"
  "5. ``(uint16 <#samples,#inputs>, float <#samples,#outputs>)`` will compute the multi-variate prediction for several feature vectors.\n"
  "6. ``(uint16 <#samples,#inputs>, float <#samples,#outputs>, float <#samples,#outputs>)`` will compute the multi-variate prediction and the labels for several feature vectors.",
  true
)
.add_prototype("features", "prediction")
.add_prototype("features, predictions")
.add_prototype("features, predictions, labels")
.add_parameter("features", "uint16 <#inputs> or uint16 <#samples, #inputs>", "The feature vector(s) the prediction should be computed for.")
.add_parameter("predictions", "float <#samples> or float <#outputs> or float <#samples, #outputs>", "The predicted values -- see below.")
.add_parameter("labels", "float <#samples> or float <#samples, #outputs>", "The predicted labels:\n\n* for the uni-variate case, -1 or +1 is assigned according to threshold 0\n* for the multi-variate case, +1 is assigned for the highest value, and 0 for all others")
.add_return("prediction", "float", "The predicted value - in case a single feature is provided and a single output is required")
;

template <int N1, int N2> void _forward(BoostedMachineObject* self, PyBlitzArrayObject* features, PyBlitzArrayObject* predictions, PyBlitzArrayObject* labels){
  const auto f = PyBlitzArrayCxx_AsBlitz<uint16_t,N1>(features);
  auto p = PyBlitzArrayCxx_AsBlitz<double,N2>(predictions);
  if (labels){
    auto l = PyBlitzArrayCxx_AsBlitz<double,N2>(labels);
    self->base->forward(*f, *p, *l);
  } else
    self->base->forward(*f, *p);
}
void _forward(BoostedMachineObject* self, PyBlitzArrayObject* features, PyBlitzArrayObject* predictions){
  const auto f = PyBlitzArrayCxx_AsBlitz<uint16_t,1>(features);
  auto p = PyBlitzArrayCxx_AsBlitz<double,1>(predictions);
  self->base->forward(*f, *p);
}


static PyObject* boostedMachine_forward(
  BoostedMachineObject* self,
  PyObject* args,
  PyObject* kwargs
)
{
  // get list of arguments
  char* kwlist[] = {c("features"), c("predictions"), c("labels"), NULL};

  PyBlitzArrayObject* p_features = 0,* p_predictions = 0,* p_labels = 0;

  if (!PyArg_ParseTupleAndKeywords(
          args, kwargs,
          "O&|O&O&", kwlist,
          &PyBlitzArray_Converter, &p_features,
          &PyBlitzArray_Converter, &p_predictions,
          &PyBlitzArray_Converter, &p_labels
      )
  )
    return NULL;

  auto _1 = make_safe(p_features), _2 = make_xsafe(p_predictions);

  try{
    if (!p_predictions){
      // uni-variate, single feature
      const auto features = PyBlitzArrayCxx_AsBlitz<uint16_t,1>(p_features, kwlist[0]);
      if (!features){
        boostedMachine_forward_doc.print_usage();
        PyErr_SetString(PyExc_TypeError, "When a single parameter is specified, only 1D arrays of type uint16 are supported.");
        return NULL;
      }
      return Py_BuildValue("d", self->base->forward(*features));
    }

    if (p_features->type_num != NPY_UINT16){
      boostedMachine_forward_doc.print_usage();
      PyErr_SetString(PyExc_TypeError, "The parameter 'features' only supports 1D or 2D arrays of type uint16");
      return NULL;
    }
    if (p_predictions->type_num != NPY_FLOAT64){
      boostedMachine_forward_doc.print_usage();
      PyErr_SetString(PyExc_TypeError, "The parameter 'predictions' only supports 1D or 2D arrays of type float");
      return NULL;
    }
    if (p_labels && (p_labels->type_num != NPY_FLOAT64 || p_labels->ndim != p_predictions->ndim)){
      boostedMachine_forward_doc.print_usage();
      PyErr_SetString(PyExc_TypeError, "The parameter 'labels' only supports 1D or 2D arrays (same as 'predictions') of type float");
      return NULL;
    }

    if (p_features->ndim == 1 && p_predictions->ndim == 1)
      _forward(self, p_features, p_predictions);
    else if (p_features->ndim == 2 && p_predictions->ndim == 1)
      _forward<2,1>(self, p_features, p_predictions, p_labels);
    else if (p_features->ndim == 2 && p_predictions->ndim == 2)
      _forward<2,2>(self, p_features, p_predictions, p_labels);
    else {
      boostedMachine_forward_doc.print_usage();
      PyErr_Format(PyExc_TypeError, "The number of dimensions of %s (%d) and %s (%d) are not supported", kwlist[0], (int)p_features->ndim, kwlist[1], (int)p_predictions->ndim);
      return NULL;
    }
    Py_RETURN_NONE;
  } catch (std::exception& ex) {
    PyErr_SetString(PyExc_RuntimeError, ex.what());
    return NULL;
  }
  catch (...) {
    PyErr_Format(PyExc_RuntimeError, "cannot create new object of type `%s' - unknown exception thrown", Py_TYPE(self)->tp_name);
    return NULL;
  }
}

static auto boostedMachine_getIndices_doc = bob::extension::FunctionDoc(
  "feature_indices",
  "Returns the feature index that will be used in this weak machine",
  NULL,
  true
)
.add_prototype("[start, [end]]", "indices")
.add_parameter("start", "int", "The first machine index to the the indices for; defaults to 0")
.add_parameter("end", "int", "The last machine index +1 to the the indices for; defaults to -1, which correspponds to the last machine + 1")
.add_return("indices", "array_like <int32>", "The feature indices required by the selected machines")
;

static PyObject* boostedMachine_getIndices(
  BoostedMachineObject* self,
  PyObject* args,
  PyObject* kwargs
)
{
  // get list of arguments
  char* kwlist[] = {c("start"), c("end"), NULL};
  int start = 0, end = -1;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|ii", kwlist, &start, &end)){
    boostedMachine_getIndices_doc.print_usage();
    return NULL;
  }

  const auto retval = self->base->getIndices(start, end);
  return PyBlitzArrayCxx_AsConstNumpy(retval);
}


static auto boostedMachine_load_doc = bob::extension::FunctionDoc(
  "load",
  "Loads the Strong machine from the given HDF5 file",
  NULL,
  true
)
.add_prototype("hdf5")
.add_parameter("hdf5", ":py:class:`bob.io.base.HDF5File`", "The HDF5 file to load this machine from.")
;

static PyObject* boostedMachine_load(
  BoostedMachineObject* self,
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
    boostedMachine_load_doc.print_usage();
    return NULL;
  }

  auto _1 = make_safe(file);
  self->base->load(*file->f);
  Py_RETURN_NONE;
}


static auto boostedMachine_save_doc = bob::extension::FunctionDoc(
  "save",
  "Saves the content of this machine to the given HDF5 file",
  NULL,
  true
)
.add_prototype("hdf5")
.add_parameter("hdf5", ":py:class:`bob.io.base.HDF5File`", "The HDF5 file to save this weak machine to.")
;

static PyObject* boostedMachine_save(
  BoostedMachineObject* self,
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
    boostedMachine_save_doc.print_usage();
    return NULL;
  }

  auto _1 = make_safe(file);
  self->base->save(*file->f);
  Py_RETURN_NONE;
}

// bind the class
static PyGetSetDef boostedMachine_Getters[] = {
  {
    boostedMachine_indices_doc.name(),
    (getter)boostedMachine_indices,
    NULL,
    boostedMachine_indices_doc.doc(),
    NULL
  },
  {
    boostedMachine_weights_doc.name(),
    (getter)boostedMachine_weights,
    NULL,
    boostedMachine_weights_doc.doc(),
    NULL
  },
  {
    c("alpha"),
    (getter)boostedMachine_weights,
    NULL,
    boostedMachine_weights_doc.doc(),
    NULL
  },
  {
    boostedMachine_outputs_doc.name(),
    (getter)boostedMachine_outputs,
    NULL,
    boostedMachine_outputs_doc.doc(),
    NULL
  },
  {
    boostedMachine_machines_doc.name(),
    (getter)boostedMachine_machines,
    NULL,
    boostedMachine_machines_doc.doc(),
    NULL
  },
  {NULL}
};

static PyMethodDef boostedMachine_Methods[] = {
  {
    boostedMachine_add_doc.name(),
    (PyCFunction)boostedMachine_add,
    METH_VARARGS | METH_KEYWORDS,
    boostedMachine_add_doc.doc(),
  },
  {
    boostedMachine_forward_doc.name(),
    (PyCFunction)boostedMachine_forward,
    METH_VARARGS | METH_KEYWORDS,
    boostedMachine_forward_doc.doc(),
  },
  {
    boostedMachine_getIndices_doc.name(),
    (PyCFunction)boostedMachine_getIndices,
    METH_VARARGS | METH_KEYWORDS,
    boostedMachine_getIndices_doc.doc(),
  },
  {
    boostedMachine_load_doc.name(),
    (PyCFunction)boostedMachine_load,
    METH_VARARGS | METH_KEYWORDS,
    boostedMachine_load_doc.doc(),
  },
  {
    boostedMachine_save_doc.name(),
    (PyCFunction)boostedMachine_save,
    METH_VARARGS | METH_KEYWORDS,
    boostedMachine_save_doc.doc(),
  },
  {NULL}
};

// Define Jesorsky Loss Type object; will be filled later
PyTypeObject BoostedMachineType = {
  PyVarObject_HEAD_INIT(0,0)
  0
};

bool init_BoostedMachine(PyObject* module)
{

  // initialize the JesorskyLossType struct
  BoostedMachineType.tp_name = boostedMachine_doc.name();
  BoostedMachineType.tp_basicsize = sizeof(BoostedMachineObject);
  BoostedMachineType.tp_flags = Py_TPFLAGS_DEFAULT;
  BoostedMachineType.tp_doc = boostedMachine_doc.doc();

  // set the functions
  BoostedMachineType.tp_new = PyType_GenericNew;
  BoostedMachineType.tp_init = reinterpret_cast<initproc>(boostedMachine_init);
  BoostedMachineType.tp_dealloc = reinterpret_cast<destructor>(boostedMachine_exit);
  BoostedMachineType.tp_call = reinterpret_cast<ternaryfunc>(boostedMachine_forward);
  BoostedMachineType.tp_getset = boostedMachine_Getters;
  BoostedMachineType.tp_methods = boostedMachine_Methods;

  // check that everyting is fine
  if (PyType_Ready(&BoostedMachineType) < 0)
    return false;

  // add the type to the module
  Py_INCREF(&BoostedMachineType);
  return PyModule_AddObject(module, boostedMachine_doc.name(), (PyObject*)&BoostedMachineType) >= 0;
}

