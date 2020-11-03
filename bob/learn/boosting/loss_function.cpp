
#include "main.h"

static auto lossFunction_doc = bob::extension::ClassDoc(
  "LossFunction",
  "Implements default Loss function behaviour.",
  "This pure virtual base class implements two functions that are required by all derived classes."
  "This class cannot be instantiated.\n\n"
  "Objects of this class are designed to be used in combination with the ``scipy.optimize.fmin_l_bfgs_b`` function."
  "Use the :py:func:`loss_sum` function as the ``func`` flag, and :py:func:`loss_gradient_sum` as ``fprime``, e.g.:\n\n"
  ".. code-block:: py\n\n"
  "   loss = bob.boosting.loss.JesorskyLoss()\n"
  "   res = scipy.optimize.fmin_l_bfgs_b(\n"
  "       func   = loss.loss_sum,\n"
  "       fprime = loss.loss_gradient_sum,\n"
  "       args   = (targets, current_strong_scores, current_weak_scores),\n"
  "       ...\n"
  "    )\n\n"
  "where ``current_strong_scores`` are the scores for the current strong machine (without the latest weak machine added) and ``current_weak_scores`` are the scores of the selected weak machine."
  "Please see the code of :py:class:`bob.boosting.trainer.Boosting` for an example."
);


static auto lossFunction_lossSum_doc = bob::extension::FunctionDoc(
  "loss_sum",
  "Computes the sum of the losses computed between the targets and the scores.",
  "This function is designed to be used with the L-BFGS method."
  "It computes the new loss based on the loss from the current strong classifier, adding the new weak machine with the currently selected weight ``alpha``",
  true
)
.add_prototype("alpha, targets, previous_scores, current_scores", "loss_sum")
.add_parameter("alpha", "float <#outputs>", "The weight for the current_scores that will be optimized in L-BFGS")
.add_parameter("targets", "float <#samples, #outputs>", "The target values that should be achieved during boosting")
.add_parameter("previous_scores", "float <#samples, #outputs>", "The score values that are achieved by the boosted machine after the previous boosting iteration")
.add_parameter("current_scores", "float <#samples, #outputs>", "The score values that are achieved with the weak machine added in this boosting round")
.add_return("loss_sum", "float <1>", "The sum over the loss values for the newly combined strong classifier")
;

static PyObject* lossFunction_lossSum(
  LossFunctionObject* self,
  PyObject* args,
  PyObject* kwargs
)
{
  // get list of arguments
  char* kwlist[] = {const_cast<char*>("alpha"), const_cast<char*>("targets"), const_cast<char*>("previous_scores"), const_cast<char*>("current_scores"), NULL};

  PyBlitzArrayObject* p_alpha = 0,* p_targets = 0,* p_prev_scores = 0,* p_curr_scores = 0;

  if (!PyArg_ParseTupleAndKeywords(
          args, kwargs,
          "O&O&O&O&", kwlist,
          &PyBlitzArray_Converter, &p_alpha,
          &PyBlitzArray_Converter, &p_targets,
          &PyBlitzArray_Converter, &p_prev_scores,
          &PyBlitzArray_Converter, &p_curr_scores)
  ){
    lossFunction_lossSum_doc.print_usage();
    return NULL;
  }

  auto _1 = make_safe(p_alpha), _2 = make_safe(p_targets), _3 = make_safe(p_prev_scores), _4 = make_safe(p_curr_scores);

  // prepare C++ data
  const auto alpha = PyBlitzArrayCxx_AsBlitz<double,1>(p_alpha, "alpha");
  const auto targets = PyBlitzArrayCxx_AsBlitz<double,2>(p_targets, "targets");
  const auto prev_scores = PyBlitzArrayCxx_AsBlitz<double,2>(p_prev_scores, "previous_scores");
  const auto curr_scores = PyBlitzArrayCxx_AsBlitz<double,2>(p_curr_scores, "current_scores");

  if (!alpha || !targets || !prev_scores || !curr_scores){
    return NULL;
  }

  blitz::Array<double,1> loss_sum(1);

  // actually call the function
  self->base->lossSum(
    *alpha,
    *targets,
    *prev_scores,
    *curr_scores,
    loss_sum
  );

  return PyBlitzArrayCxx_AsNumpy(loss_sum);
}


static auto lossFunction_gradientSum_doc = bob::extension::FunctionDoc(
  "loss_gradient_sum",
  "Computes the sum of the loss gradients computed between the targets and the scores.",
  "This function is designed to be used with the L-BFGS method."
  "It computes the new derivative of the loss based on the loss from the current strong classifier, adding the new weak machine with the currently selected weight ``alpha``" ,
  true
)
.add_prototype("alpha, targets, previous_scores, current_scores", "gradient_sum")
.add_parameter("alpha", "float <#outputs>", "The weight for the current_scores that will be optimized in L-BFGS")
.add_parameter("targets", "float <#samples, #outputs>", "The target values that should be achieved during boosting")
.add_parameter("previous_scores", "float <#samples, #outputs>", "The score values that are achieved by the boosted machine after the previous boosting iteration")
.add_parameter("current_scores", "float <#samples, #outputs>", "The score values that are achieved with the weak machine added in this boosting round")
.add_return("gradient_sum", "float <#outputs>", "The sum over the loss gradients for the newly combined strong classifier")
;

static PyObject* lossFunction_gradientSum(
  LossFunctionObject* self,
  PyObject* args,
  PyObject* kwargs
)
{
  // get list of arguments
  char* kwlist[] = {const_cast<char*>("alpha"), const_cast<char*>("targets"), const_cast<char*>("previous_scores"), const_cast<char*>("current_scores"), NULL};

  PyBlitzArrayObject* p_alpha = 0,* p_targets = 0,* p_prev_scores = 0,* p_curr_scores = 0;

  if (!PyArg_ParseTupleAndKeywords(
          args, kwargs,
          "O&O&O&O&", kwlist,
          &PyBlitzArray_Converter, &p_alpha,
          &PyBlitzArray_Converter, &p_targets,
          &PyBlitzArray_Converter, &p_prev_scores,
          &PyBlitzArray_Converter, &p_curr_scores)
  ){
    lossFunction_gradientSum_doc.print_usage();
    return NULL;
  }

  auto _1 = make_safe(p_alpha), _2 = make_safe(p_targets), _3 = make_safe(p_prev_scores), _4 = make_safe(p_curr_scores);

  // prepare C++ data
  const auto alpha = PyBlitzArrayCxx_AsBlitz<double,1>(p_alpha, "alpha");
  const auto targets = PyBlitzArrayCxx_AsBlitz<double,2>(p_targets, "targets");
  const auto prev_scores = PyBlitzArrayCxx_AsBlitz<double,2>(p_prev_scores, "previous_scores");
  const auto curr_scores = PyBlitzArrayCxx_AsBlitz<double,2>(p_curr_scores, "current_scores");

  if (!alpha || !targets || !prev_scores || !curr_scores){
    return NULL;
  }

  blitz::Array<double,1> gradient_sum(targets->extent(1));

  // actually call the function
  self->base->gradientSum(
    *alpha,
    *targets,
    *prev_scores,
    *curr_scores,
    gradient_sum
  );

  return PyBlitzArrayCxx_AsNumpy(gradient_sum);
}

// bind the class
static PyMethodDef lossFunction_Methods[] = {
  {
    lossFunction_lossSum_doc.name(),
    (PyCFunction)lossFunction_lossSum,
    METH_VARARGS | METH_KEYWORDS,
    lossFunction_lossSum_doc.doc(),
  },
  {
    lossFunction_gradientSum_doc.name(),
    (PyCFunction)lossFunction_gradientSum,
    METH_VARARGS | METH_KEYWORDS,
    lossFunction_gradientSum_doc.doc(),
  },
  {NULL}
};


PyTypeObject LossFunctionType = {
  PyVarObject_HEAD_INIT(0,0)
  0
};

bool init_LossFunction(PyObject* module)
{

  // initialize the JesorskyLossType struct
  LossFunctionType.tp_name = lossFunction_doc.name();
  LossFunctionType.tp_basicsize = sizeof(LossFunctionObject);
  LossFunctionType.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
  LossFunctionType.tp_doc = lossFunction_doc.doc();

  // set the functions
  LossFunctionType.tp_methods = lossFunction_Methods;

  // check that everyting is fine
  if (PyType_Ready(&LossFunctionType) < 0)
    return false;

  // add the type to the module
  Py_INCREF(&LossFunctionType);
  return PyModule_AddObject(module, lossFunction_doc.name(), (PyObject*)&LossFunctionType) >= 0;
}
