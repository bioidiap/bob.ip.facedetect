
#include "main.h"

static auto jesorskyLoss_doc = bob::extension::ClassDoc(
  "JesorskyLoss",
  "Computes the Jesorsky loss and its derivative.",
  "The Jesorsky loss defines an error function between the target vectors and the currently achieved scores."
  "It is specifically designed to perform regression in a facial feature localization (FFL) task."
  "It assumes that the feature vector :math:`\\vec p` consists of facial landmark positions, which are given in the following way:\n\n"
  ".. math:: \\vec p = [y_0, x_0, y_1, x_1, \\dots, y_{n-1}, x_{n-1}]\n\n"
  "with :math:`(y_0, x_0)` is the **right eye landmark**, :math:`(y_0, x_0)` is the **left eye landmark** and all other landmarks (a total of :math:`n` landmarks) are following.\n\n"
  "The error between target vector :math:`\\vec a` and test vector :math:`\\vec b` is computed as the average landmark-wise Euclidean distance, normalized by the inter-eye-distance of the target vector:\n\n"
  ".. math:: d(\\vec a, \\vec b) = \\sum_{i=0}^n \\frac{\\sqrt{(b_{2i} - a_{2i})^2 + (b_{2i+1} - a_{2i+1})^2}} {\\sqrt{(a_0 - a_2)^2 + (a_1 - a_3)^2}}\n\n"
  "The derivative is a bit more complicated."
  "First, the error is computed for each landmark :math:`i=0,\\dots,n-1`:\n\n"
  ".. math:: d_i(\\vec a, \\vec b) = \\frac{\\sqrt{(b_{2i} - a_{2i})^2 + (b_{2i+1} - a_{2i+1})^2}} {\\sqrt{(a_0 - a_2)^2 + (a_1 - a_3)^2}}\n\n"
  "and then the derivative is computed for each element of the target vector:\n\n"
  ".. math:: \\nabla(\\vec a, \\vec b) = \\left[d_i\\cdot(b_{2i} - a_{2i}), d_i\\cdot(b_{2i+1} - a_{2i+1}) \\right]_i \n\n"

)
.add_constructor(
  bob::extension::FunctionDoc(
    "__init__",
    "Initializes a JesorskyLoss object.",
    "The constructor comes with no parameters.",
    true
  ).add_prototype("", "")
);



// Some functions
static int jesorskyLoss_init(
  JesorskyLossObject* self,
  PyObject* args,
  PyObject* kwargs
)
{
  // get list of arguments
  char* kwlist[] = {NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "", kwlist)) return -1;

  self->base.reset(new bob::learn::boosting::JesorskyLoss());
  self->parent.base = self->base;
  return 0;
}

static void jesorskyLoss_exit(
  JesorskyLossObject* self
)
{
  self->base.reset();
  self->parent.base.reset();
  Py_TYPE(self)->tp_free(reinterpret_cast<PyObject*>(self));
}


static auto jesorskyLoss_loss_doc = bob::extension::FunctionDoc(
  "loss",
  "Computes the Jesorsky error between the targets and the scores.",
  "This function computes the Jesorsky error between all given targets and samples, using the loss formula as explained above :py:class:`JesorskyLoss`",
  true
)
.add_prototype("targets, scores", "errors")
.add_parameter("targets", "float <#samples, #outputs>", "The target values that should be achieved during boosting")
.add_parameter("scores", "float <#samples, #outputs>", "The score values that are currently achieved")
.add_return("errors", "float <#samples, 1>", "The resulting Jesorsky errors for each target")
;

static PyObject* jesorskyLoss_loss(
  JesorskyLossObject* self,
  PyObject* args,
  PyObject* kwargs
)
{
  // get list of arguments
  char* kwlist[] = {const_cast<char*>("targets"), const_cast<char*>("scores"), NULL};

  PyBlitzArrayObject* p_targets = 0,* p_scores = 0;

  if (!PyArg_ParseTupleAndKeywords(
          args, kwargs,
          "O&O&", kwlist,
          &PyBlitzArray_Converter, &p_targets,
          &PyBlitzArray_Converter, &p_scores)
  ){
    jesorskyLoss_loss_doc.print_usage();
    return NULL;
  }

  auto _1 = make_safe(p_targets), _2 = make_safe(p_scores);

  // prepare C++ data
  const auto targets = PyBlitzArrayCxx_AsBlitz<double,2>(p_targets, "targets");
  const auto scores = PyBlitzArrayCxx_AsBlitz<double,2>(p_scores, "scores");

  if (!targets || !scores){
    return NULL;
  }

  blitz::Array<double,2> errors(targets->extent(0), 1);

  // actually call the function
  self->base->loss(
    *targets,
    *scores,
    errors
  );

  return PyBlitzArrayCxx_AsNumpy(errors);
}


static auto jesorskyLoss_lossGradient_doc = bob::extension::FunctionDoc(
  "loss_gradient",
  "Computes the Jesorsky error between the targets and the scores.",
  "This function computes the derivative of the Jesorsky error between all given targets and samples, using the loss formula as explained above :py:class:`JesorskyLoss`",
  true
)
.add_prototype("targets, scores", "gradient")
.add_parameter("targets", "float <#samples, #outputs>", "The target values that should be achieved during boosting")
.add_parameter("scores", "float <#samples, #outputs>", "The score values that are currently achieved")
.add_return("gradient", "float <#samples, #outputs>", "The derivative of the Jesorsky error for each sample")
;

static PyObject* jesorskyLoss_lossGradient(
  JesorskyLossObject* self,
  PyObject* args,
  PyObject* kwargs
)
{
  // get list of arguments
  char* kwlist[] = {const_cast<char*>("targets"), const_cast<char*>("scores"), NULL};

  PyBlitzArrayObject* p_targets = 0,* p_scores = 0;

  if (!PyArg_ParseTupleAndKeywords(
          args, kwargs,
          "O&O&", kwlist,
          &PyBlitzArray_Converter, &p_targets,
          &PyBlitzArray_Converter, &p_scores)
  ){
    jesorskyLoss_lossGradient_doc.print_usage();
    return NULL;
  }

  auto _1 = make_safe(p_targets), _2 = make_safe(p_scores);

  // prepare C++ data
  const auto targets = PyBlitzArrayCxx_AsBlitz<double,2>(p_targets, "targets");
  const auto scores = PyBlitzArrayCxx_AsBlitz<double,2>(p_scores, "scores");

  if (!targets || !scores)
    return NULL;

  blitz::Array<double,2> gradient(targets->shape());

  // actually call the function
  self->base->lossGradient(
    *targets,
    *scores,
    gradient
  );

  return PyBlitzArrayCxx_AsNumpy(gradient);
}

// bind the class
static PyMethodDef jesorskyLoss_Methods[] = {
  {
    jesorskyLoss_loss_doc.name(),
    (PyCFunction)jesorskyLoss_loss,
    METH_VARARGS | METH_KEYWORDS,
    jesorskyLoss_loss_doc.doc(),
  },
  {
    jesorskyLoss_lossGradient_doc.name(),
    (PyCFunction)jesorskyLoss_lossGradient,
    METH_VARARGS | METH_KEYWORDS,
    jesorskyLoss_lossGradient_doc.doc(),
  },
  {NULL}
};


// Define Jesorsky Loss Type object; will be filled later
PyTypeObject JesorskyLossType = {
  PyVarObject_HEAD_INIT(0,0)
  0
};

bool init_JesorskyLoss(PyObject* module)
{

  // initialize the JesorskyLossType struct
  JesorskyLossType.tp_name = jesorskyLoss_doc.name();
  JesorskyLossType.tp_basicsize = sizeof(JesorskyLossObject);
  JesorskyLossType.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
  JesorskyLossType.tp_doc = jesorskyLoss_doc.doc();
  JesorskyLossType.tp_base = &LossFunctionType;

  // set the functions
  JesorskyLossType.tp_new = PyType_GenericNew;
  JesorskyLossType.tp_init = reinterpret_cast<initproc>(jesorskyLoss_init);
  JesorskyLossType.tp_dealloc = reinterpret_cast<destructor>(jesorskyLoss_exit);
  JesorskyLossType.tp_methods = jesorskyLoss_Methods;

  // check that everyting is fine
  if (PyType_Ready(&JesorskyLossType) < 0)
    return false;

  // add the type to the module
  Py_INCREF(&JesorskyLossType);
  return PyModule_AddObject(module, jesorskyLoss_doc.name(), (PyObject*)&JesorskyLossType) >= 0;
}
