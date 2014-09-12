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

static auto BoundingBox_doc = bob::extension::ClassDoc(
  BOB_EXT_MODULE_PREFIX ".BoundingBox",
  "A bounding box class storing top, left, height and width of an rectangle",
  0
).add_constructor(
  bob::extension::FunctionDoc(
    "__init__",
    "Constructs a new Bounding box from the given top-left position and the size of the rectangle",
    0,
    true
  )
  .add_prototype("topleft, size", "")
  .add_prototype("bounding_box", "")
  .add_parameter("topleft", "(float, float)", "The top-left position of the bounding box")
  .add_parameter("size", "(float, float)", "The size of the bounding box")
  .add_parameter("bounding_box", ":py:class:`BoundingBox`", "The BoundingBox object to use for copy-construction")
);


static int BoundingBox_init(BoundingBoxObject* self, PyObject* args, PyObject* kwargs) {
  TRY

  char* kwlist1[] = {c("topleft"), c("size"), NULL};
  char* kwlist2[] = {c("bounding_box"), NULL};

  // get the number of command line arguments
  Py_ssize_t nargs = (args?PyTuple_Size(args):0) + (kwargs?PyDict_Size(kwargs):0);

  if (nargs == 1){
    // copy construct
    BoundingBoxObject* bb;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!", kwlist2, &BoundingBox_Type, &bb)) return -1;

    self->cxx.reset(new BoundingBox(*bb->cxx));
    return 0;
  }

  blitz::TinyVector<double,2> topleft, size;
  if (!(PyArg_ParseTupleAndKeywords(args, kwargs, "(dd)|(dd)", kwlist1, &topleft[0], &topleft[1], &size[0], &size[1]))) return -1;
  self->cxx.reset(new BoundingBox(topleft[0], topleft[1], size[0], size[1]));
  return 0;

  CATCH("cannot create BoundingBox", -1)
}

static void BoundingBox_delete(BoundingBoxObject* self) {
  Py_TYPE(self)->tp_free((PyObject*)self);
}

int BoundingBox_Check(PyObject* o) {
  return PyObject_IsInstance(o, reinterpret_cast<PyObject*>(&BoundingBox_Type));
}

static PyObject* BoundingBox_RichCompare(BoundingBoxObject* self, PyObject* other, int op) {
  TRY
  if (!BoundingBox_Check(other)) {
    PyErr_Format(PyExc_TypeError, "cannot compare `%s' with `%s'", Py_TYPE(self)->tp_name, Py_TYPE(other)->tp_name);
    return 0;
  }
  auto other_ = reinterpret_cast<BoundingBoxObject*>(other);
  switch (op) {
    case Py_EQ:
      if (*self->cxx==*other_->cxx) Py_RETURN_TRUE; else Py_RETURN_FALSE;
    case Py_NE:
      if (*self->cxx==*other_->cxx) Py_RETURN_FALSE; else Py_RETURN_TRUE;
    default:
      Py_INCREF(Py_NotImplemented);
      return Py_NotImplemented;
  }
  CATCH("cannot compare BoundingBox objects", 0)
}

PyObject* BoundingBox_Str(BoundingBoxObject* self) {
  TRY
  return PyString_FromString((boost::format("<BB topleft=(%3.2d,%3.2d), bottomright=(%3.2d,%3.2d)>") % self->cxx->top() % self->cxx->left() % self->cxx->bottom() % self->cxx->right()).str().c_str());
  CATCH("cannot create __repr__ string", 0)
}


/******************************************************************/
/************ Variables Section ***********************************/
/******************************************************************/

static auto topleft = bob::extension::VariableDoc(
  "topleft",
  "(int, int)",
  "The top-left position of the bounding box as integral values, read access only"
);
PyObject* BoundingBox_topleft(BoundingBoxObject* self, void*){
  TRY
  return Py_BuildValue("ii", self->cxx->itop(), self->cxx->ileft());
  CATCH("topleft could not be read", 0)
}

static auto bottomright = bob::extension::VariableDoc(
  "bottomright",
  "(int, int)",
  "The bottom-right position of the bounding box as integral values, read access only"
);
PyObject* BoundingBox_bottomright(BoundingBoxObject* self, void*){
  TRY
  return Py_BuildValue("ii", self->cxx->ibottom(), self->cxx->iright());
  CATCH("bottomright could not be read", 0)
}

static auto size = bob::extension::VariableDoc(
  "size",
  "(int, int)",
  "The size of the bounding box as integral values, read access only"
);
PyObject* BoundingBox_size(BoundingBoxObject* self, void*){
  TRY
  return Py_BuildValue("ii", self->cxx->iheight(), self->cxx->iwidth());
  CATCH("topleft could not be read", 0)
}


static auto topleft_f = bob::extension::VariableDoc(
  "topleft_f",
  "(float, float)",
  "The top-left position of the bounding box as float values, read access only"
);
PyObject* BoundingBox_topleft_f(BoundingBoxObject* self, void*){
  TRY
  return Py_BuildValue("dd", self->cxx->top(), self->cxx->left());
  CATCH("topleft_f could not be read", 0)
}

static auto bottomright_f = bob::extension::VariableDoc(
  "bottomright_f",
  "(float, float)",
  "The bottom-right position of the bounding box as float values, read access only"
);
PyObject* BoundingBox_bottomright_f(BoundingBoxObject* self, void*){
  TRY
  return Py_BuildValue("dd", self->cxx->bottom(), self->cxx->right());
  CATCH("bottomright_f could not be read", 0)
}

static auto size_f = bob::extension::VariableDoc(
  "size_f",
  "(float, float)",
  "The size of the bounding box as float values, read access only"
);
PyObject* BoundingBox_size_f(BoundingBoxObject* self, void*){
  TRY
  return Py_BuildValue("dd", self->cxx->height(), self->cxx->width());
  CATCH("size_f could not be read", 0)
}


static auto center = bob::extension::VariableDoc(
  "center",
  "(float, float)",
  "The center of the bounding box (as float values), read access only"
);
PyObject* BoundingBox_center(BoundingBoxObject* self, void*){
  TRY
  return Py_BuildValue("dd", self->cxx->center()[0], self->cxx->center()[1]);
  CATCH("center could not be read", 0)
}

static auto area = bob::extension::VariableDoc(
  "area",
  "float",
  "The area (height x width) of the bounding box, read access only"
);
PyObject* BoundingBox_area(BoundingBoxObject* self, void*){
  TRY
  return Py_BuildValue("d", self->cxx->area());
  CATCH("area could not be read", 0)
}

static PyGetSetDef BoundingBox_getseters[] = {
    {
      topleft.name(),
      (getter)BoundingBox_topleft,
      0,
      topleft.doc(),
      0
    },
    {
      topleft_f.name(),
      (getter)BoundingBox_topleft_f,
      0,
      topleft_f.doc(),
      0
    },
    {
      bottomright.name(),
      (getter)BoundingBox_bottomright,
      0,
      bottomright.doc(),
      0
    },
    {
      bottomright_f.name(),
      (getter)BoundingBox_bottomright_f,
      0,
      bottomright_f.doc(),
      0
    },
    {
      size.name(),
      (getter)BoundingBox_size,
      0,
      size.doc(),
      0
    },
    {
      size_f.name(),
      (getter)BoundingBox_size_f,
      0,
      size_f.doc(),
      0
    },
    {
      center.name(),
      (getter)BoundingBox_center,
      0,
      center.doc(),
      0
    },
    {
      area.name(),
      (getter)BoundingBox_area,
      0,
      area.doc(),
      0
    },
    {0}  /* Sentinel */
};

/******************************************************************/
/************ Functions Section ***********************************/
/******************************************************************/

static auto shift = bob::extension::FunctionDoc(
  "shift",
  "This function returns a shifted version of this BoundingBox",
  0,
  true
)
.add_prototype("offset", "bounding_box")
.add_parameter("offset", "(float, float)", "The offset with which this bounding box should be shifted")
.add_return("bounding_box", ":py:class:`BoundingBox`", "The shifted version of this bounding box")
;

static PyObject* BoundingBox_shift(BoundingBoxObject* self, PyObject* args, PyObject* kwargs) {
  TRY
  static char* kwlist[] = {c("offset"), 0};

  blitz::TinyVector<double,2> offset;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "(dd)", kwlist, &offset[0], &offset[1])) return 0;
  BoundingBoxObject* ret = reinterpret_cast<BoundingBoxObject*>(BoundingBox_Type.tp_alloc(&BoundingBox_Type, 0));
  ret->cxx = self->cxx->shift(offset[0], offset[1]);

  return (PyObject*)ret;
//  return Py_BuildValue("N", ret);
  CATCH("cannot shift", 0)
}

static auto scale = bob::extension::FunctionDoc(
  "scale",
  "This function returns a scaled version of this BoundingBox",
  "When the ``centered`` parameter is set to ``True``, the transformation center will be in the center of this bounding box, otherwise it will be at (0,0)",
  true
)
.add_prototype("scale, [centered]", "bounding_box")
.add_parameter("scale", "float", "The scale with which this bounding box should be shifted")
.add_parameter("centered", "bool", "[Default: ``False``] : Should the scaling done with repect to the center of the bounding box?")
.add_return("bounding_box", ":py:class:`BoundingBox`", "The scaled version of this bounding box")
;

static PyObject* BoundingBox_scale(BoundingBoxObject* self, PyObject* args, PyObject* kwargs) {
  TRY
  static char* kwlist[] = {c("scale"), c("centered"), 0};

  double scale;
  PyObject* centered = 0;
  // by shape
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "d|O!", kwlist, &scale, &PyBool_Type, &centered)){
    return 0;
  }
  BoundingBoxObject* ret = reinterpret_cast<BoundingBoxObject*>(BoundingBox_Type.tp_alloc(&BoundingBox_Type, 0));
  if (f(centered))
    ret->cxx = self->cxx->scaleCentered(scale);
  else
    ret->cxx = self->cxx->scale(scale);
  return Py_BuildValue("N", ret);
  CATCH("cannot scale", 0)
}

static auto mirror_x = bob::extension::FunctionDoc(
  "mirror_x",
  "This function returns a horizontally mirrored version of this BoundingBox",
  0,
  true
)
.add_prototype("width", "bounding_box")
.add_parameter("width", "int", "The width of the image at which this bounding box should be mirrored")
.add_return("bounding_box", ":py:class:`BoundingBox`", "The mirrored version of this bounding box")
;
static PyObject* BoundingBox_mirror_x(BoundingBoxObject* self, PyObject* args, PyObject* kwargs) {
  TRY
  static char* kwlist[] = {c("width"), 0};

  int width;
  // by shape
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "i", kwlist, &width)){
    return 0;
  }
  BoundingBoxObject* ret = reinterpret_cast<BoundingBoxObject*>(BoundingBox_Type.tp_alloc(&BoundingBox_Type, 0));
  ret->cxx = self->cxx->mirrorX(width);
  return Py_BuildValue("N", ret);
  CATCH("cannot mirror horizontally", 0)
}

static auto overlap = bob::extension::FunctionDoc(
  "overlap",
  "This function returns the overlapping bounding box between this and the given bounding box",
  0,
  true
)
.add_prototype("other", "bounding_box")
.add_parameter("other", ":py:class:`BoundingBox`", "The other bounding box to compute the overlap with")
.add_return("bounding_box", ":py:class:`BoundingBox`", "The overlap between this and the other bounding box")
;
static PyObject* BoundingBox_overlap(BoundingBoxObject* self, PyObject* args, PyObject* kwargs) {
  TRY
  static char* kwlist[] = {c("other"), 0};

  BoundingBoxObject* other;
  // by shape
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!", kwlist, &BoundingBox_Type, &other)) return 0;

  BoundingBoxObject* ret = reinterpret_cast<BoundingBoxObject*>(BoundingBox_Type.tp_alloc(&BoundingBox_Type, 0));
  ret->cxx = self->cxx->overlap(*other->cxx);
  return Py_BuildValue("N", ret);
  CATCH("cannot compute overlap", 0)
}

static auto similarity = bob::extension::FunctionDoc(
  "similarity",
  "This function computes the Jaccard similarity index between this and the given BoundingBox",
  0,
  true
)
.add_prototype("other", "sim")
.add_parameter("other", ":py:class:`BoundingBox`", "The other bounding box to compute the overlap with")
.add_return("sim", "float", "The Jaccard similarity index between this and the given BoundingBox")
;
static PyObject* BoundingBox_similarity(BoundingBoxObject* self, PyObject* args, PyObject* kwargs) {
  TRY
  static char* kwlist[] = {c("other"), 0};

  BoundingBoxObject* other;
  // by shape
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!", kwlist, &BoundingBox_Type, &other)){
    return 0;
  }

  double sim = self->cxx->similarity(*other->cxx);
  return Py_BuildValue("d", sim);
  CATCH("cannot compute overlap", 0)
}

static auto is_valid_for = bob::extension::FunctionDoc(
  "is_valid_for",
  "Checks if the bounding box is inside the given image size",
  0,
  true
)
.add_prototype("size", "valid")
.add_parameter("size", "(int, int)", "The size of the image to test")
.add_return("valid", "bool", "``True`` if the bounding box is inside the image boundaries, ``False`` otherwise")
;
static PyObject* BoundingBox_is_valid_for(BoundingBoxObject* self, PyObject* args, PyObject* kwargs) {
  TRY
  static char* kwlist[] = {c("size"), 0};

  blitz::TinyVector<int,2> size;
  // by shape
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "(ii)", kwlist, &size[0], &size[1])){
    return 0;
  }
  if (self->cxx->isValidFor(size))
    Py_RETURN_TRUE;
  else
    Py_RETURN_FALSE;
  CATCH("cannot compute validity", 0)
}


static PyMethodDef BoundingBox_methods[] = {
  {
    scale.name(),
    (PyCFunction)BoundingBox_scale,
    METH_VARARGS|METH_KEYWORDS,
    scale.doc()
  },
  {
    shift.name(),
    (PyCFunction)BoundingBox_shift,
    METH_VARARGS|METH_KEYWORDS,
    shift.doc()
  },
  {
    mirror_x.name(),
    (PyCFunction)BoundingBox_mirror_x,
    METH_VARARGS|METH_KEYWORDS,
    mirror_x.doc()
  },
  {
    overlap.name(),
    (PyCFunction)BoundingBox_overlap,
    METH_VARARGS|METH_KEYWORDS,
    overlap.doc()
  },
  {
    similarity.name(),
    (PyCFunction)BoundingBox_similarity,
    METH_VARARGS|METH_KEYWORDS,
    similarity.doc()
  },
  {
    is_valid_for.name(),
    (PyCFunction)BoundingBox_is_valid_for,
    METH_VARARGS|METH_KEYWORDS,
    is_valid_for.doc()
  },
  {0} /* Sentinel */
};


/******************************************************************/
/************ Module Section **************************************/
/******************************************************************/

// Define the DCTFeatures type struct; will be initialized later
PyTypeObject BoundingBox_Type = {
  PyVarObject_HEAD_INIT(0,0)
  0
};

bool init_BoundingBox(PyObject* module)
{
  // initialize the type struct
  BoundingBox_Type.tp_name = BoundingBox_doc.name();
  BoundingBox_Type.tp_basicsize = sizeof(BoundingBoxObject);
  BoundingBox_Type.tp_flags = Py_TPFLAGS_DEFAULT;
  BoundingBox_Type.tp_doc = BoundingBox_doc.doc();
  BoundingBox_Type.tp_repr = (reprfunc)BoundingBox_Str;
  BoundingBox_Type.tp_str = (reprfunc)BoundingBox_Str;

  // set the functions
  BoundingBox_Type.tp_new = PyType_GenericNew;
  BoundingBox_Type.tp_init = reinterpret_cast<initproc>(BoundingBox_init);
  BoundingBox_Type.tp_dealloc = reinterpret_cast<destructor>(BoundingBox_delete);
  BoundingBox_Type.tp_richcompare = reinterpret_cast<richcmpfunc>(BoundingBox_RichCompare);
  BoundingBox_Type.tp_methods = BoundingBox_methods;
  BoundingBox_Type.tp_getset = BoundingBox_getseters;

  // check that everything is fine
  if (PyType_Ready(&BoundingBox_Type) < 0) return false;

  // add the type to the module
  Py_INCREF(&BoundingBox_Type);
  return PyModule_AddObject(module, "BoundingBox", (PyObject*)&BoundingBox_Type) >= 0;
}

