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


static int PyBobIpFacedetectBoundingBox_init(PyBobIpFacedetectBoundingBoxObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY

  char** kwlist1 = BoundingBox_doc.kwlist(0);
  char** kwlist2 = BoundingBox_doc.kwlist(1);

  // get the number of command line arguments
  Py_ssize_t nargs = (args?PyTuple_Size(args):0) + (kwargs?PyDict_Size(kwargs):0);

  if (nargs == 1){
    // copy construct
    PyBobIpFacedetectBoundingBoxObject* bb;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!", kwlist2, &PyBobIpFacedetectBoundingBox_Type, &bb)) return -1;

    self->cxx.reset(new bob::ip::facedetect::BoundingBox(*bb->cxx));
    return 0;
  }

  blitz::TinyVector<double,2> topleft, size;
  if (!(PyArg_ParseTupleAndKeywords(args, kwargs, "(dd)(dd)", kwlist1, &topleft[0], &topleft[1], &size[0], &size[1]))) return -1;
  self->cxx.reset(new bob::ip::facedetect::BoundingBox(topleft[0], topleft[1], size[0], size[1]));
  return 0;

  BOB_CATCH_MEMBER("cannot create BoundingBox", -1)
}

static void PyBobIpFacedetectBoundingBox_delete(PyBobIpFacedetectBoundingBoxObject* self) {
  self->cxx.reset();
  Py_TYPE(self)->tp_free((PyObject*)self);
}

int PyBobIpFacedetectBoundingBox_Check(PyObject* o) {
  return PyObject_IsInstance(o, reinterpret_cast<PyObject*>(&PyBobIpFacedetectBoundingBox_Type));
}

static PyObject* PyBobIpFacedetectBoundingBox_RichCompare(PyBobIpFacedetectBoundingBoxObject* self, PyObject* other, int op) {
  BOB_TRY
  if (!PyBobIpFacedetectBoundingBox_Check(other)) {
    PyErr_Format(PyExc_TypeError, "cannot compare `%s' with `%s'", Py_TYPE(self)->tp_name, Py_TYPE(other)->tp_name);
    return 0;
  }
  auto other_ = reinterpret_cast<PyBobIpFacedetectBoundingBoxObject*>(other);
  switch (op) {
    case Py_EQ:
      if (*self->cxx==*other_->cxx) Py_RETURN_TRUE; else Py_RETURN_FALSE;
    case Py_NE:
      if (*self->cxx==*other_->cxx) Py_RETURN_FALSE; else Py_RETURN_TRUE;
    default:
      Py_INCREF(Py_NotImplemented);
      return Py_NotImplemented;
  }
  BOB_CATCH_MEMBER("cannot compare BoundingBox objects", 0)
}

PyObject* PyBobIpFacedetectBoundingBox_Str(PyBobIpFacedetectBoundingBoxObject* self) {
  BOB_TRY
  return PyString_FromString((boost::format("<BB topleft=(%3.2f, %3.2f), bottomright=(%3.2f, %3.2f)>") % self->cxx->top() % self->cxx->left() % self->cxx->bottom() % self->cxx->right()).str().c_str());
  BOB_CATCH_MEMBER("cannot create __repr__ string", 0)
}


/******************************************************************/
/************ Variables Section ***********************************/
/******************************************************************/

static auto topleft = bob::extension::VariableDoc(
  "topleft",
  "(int, int)",
  "The top-left position of the bounding box as integral values, read access only"
);
PyObject* PyBobIpFacedetectBoundingBox_topleft(PyBobIpFacedetectBoundingBoxObject* self, void*){
  BOB_TRY
  return Py_BuildValue("ii", self->cxx->itop(), self->cxx->ileft());
  BOB_CATCH_MEMBER("topleft could not be read", 0)
}

static auto bottomright = bob::extension::VariableDoc(
  "bottomright",
  "(int, int)",
  "The bottom-right position of the bounding box (which is just outside the bounding box) as integral values, read access only"
);
PyObject* PyBobIpFacedetectBoundingBox_bottomright(PyBobIpFacedetectBoundingBoxObject* self, void*){
  BOB_TRY
  return Py_BuildValue("ii", self->cxx->ibottom(), self->cxx->iright());
  BOB_CATCH_MEMBER("bottomright could not be read", 0)
}

static auto top = bob::extension::VariableDoc(
  "top",
  "int",
  "The top position of the bounding box as int, read access only"
);
PyObject* PyBobIpFacedetectBoundingBox_top(PyBobIpFacedetectBoundingBoxObject* self, void*){
  BOB_TRY
  return Py_BuildValue("i", self->cxx->itop());
  BOB_CATCH_MEMBER("top could not be read", 0)
}

static auto left = bob::extension::VariableDoc(
  "left",
  "int",
  "The left position of the bounding box as int, read access only"
);
PyObject* PyBobIpFacedetectBoundingBox_left(PyBobIpFacedetectBoundingBoxObject* self, void*){
  BOB_TRY
  return Py_BuildValue("i", self->cxx->ileft());
  BOB_CATCH_MEMBER("left could not be read", 0)
}

static auto bottom = bob::extension::VariableDoc(
  "bottom",
  "int",
  "The bottom position of the bounding box (which is just outside the bounding box) as int, read access only"
);
PyObject* PyBobIpFacedetectBoundingBox_bottom(PyBobIpFacedetectBoundingBoxObject* self, void*){
  BOB_TRY
  return Py_BuildValue("i", self->cxx->ibottom());
  BOB_CATCH_MEMBER("bottom_f could not be read", 0)
}

static auto right = bob::extension::VariableDoc(
  "right",
  "int",
  "The right position of the bounding box (which is just outside the bounding box) as int, read access only"
);
PyObject* PyBobIpFacedetectBoundingBox_right(PyBobIpFacedetectBoundingBoxObject* self, void*){
  BOB_TRY
  return Py_BuildValue("i", self->cxx->iright());
  BOB_CATCH_MEMBER("right_f could not be read", 0)
}


static auto size = bob::extension::VariableDoc(
  "size",
  "(int, int)",
  "The size of the bounding box as integral values, read access only"
);
PyObject* PyBobIpFacedetectBoundingBox_size(PyBobIpFacedetectBoundingBoxObject* self, void*){
  BOB_TRY
  return Py_BuildValue("ii", self->cxx->iheight(), self->cxx->iwidth());
  BOB_CATCH_MEMBER("topleft could not be read", 0)
}


static auto topleft_f = bob::extension::VariableDoc(
  "topleft_f",
  "(float, float)",
  "The top-left position of the bounding box as float values, read access only"
);
PyObject* PyBobIpFacedetectBoundingBox_topleft_f(PyBobIpFacedetectBoundingBoxObject* self, void*){
  BOB_TRY
  return Py_BuildValue("dd", self->cxx->top(), self->cxx->left());
  BOB_CATCH_MEMBER("topleft_f could not be read", 0)
}

static auto bottomright_f = bob::extension::VariableDoc(
  "bottomright_f",
  "(float, float)",
  "The bottom-right position of the bounding box (which is just outside the bounding box) as float values, read access only"
);
PyObject* PyBobIpFacedetectBoundingBox_bottomright_f(PyBobIpFacedetectBoundingBoxObject* self, void*){
  BOB_TRY
  return Py_BuildValue("dd", self->cxx->bottom(), self->cxx->right());
  BOB_CATCH_MEMBER("bottomright_f could not be read", 0)
}

static auto top_f = bob::extension::VariableDoc(
  "top_f",
  "float",
  "The top position of the bounding box as float, read access only"
);
PyObject* PyBobIpFacedetectBoundingBox_top_f(PyBobIpFacedetectBoundingBoxObject* self, void*){
  BOB_TRY
  return Py_BuildValue("d", self->cxx->top());
  BOB_CATCH_MEMBER("top_f could not be read", 0)
}

static auto left_f = bob::extension::VariableDoc(
  "left_f",
  "float",
  "The left position of the bounding box as float, read access only"
);
PyObject* PyBobIpFacedetectBoundingBox_left_f(PyBobIpFacedetectBoundingBoxObject* self, void*){
  BOB_TRY
  return Py_BuildValue("d", self->cxx->left());
  BOB_CATCH_MEMBER("left_f could not be read", 0)
}

static auto bottom_f = bob::extension::VariableDoc(
  "bottom_f",
  "float",
  "The bottom position of the bounding box (which is just outside the bounding box) as float, read access only"
);
PyObject* PyBobIpFacedetectBoundingBox_bottom_f(PyBobIpFacedetectBoundingBoxObject* self, void*){
  BOB_TRY
  return Py_BuildValue("d", self->cxx->bottom());
  BOB_CATCH_MEMBER("bottom_f could not be read", 0)
}

static auto right_f = bob::extension::VariableDoc(
  "right_f",
  "float",
  "The right position of the bounding box (which is just outside the bounding box) as float, read access only"
);
PyObject* PyBobIpFacedetectBoundingBox_right_f(PyBobIpFacedetectBoundingBoxObject* self, void*){
  BOB_TRY
  return Py_BuildValue("d", self->cxx->right());
  BOB_CATCH_MEMBER("right_f could not be read", 0)
}

static auto size_f = bob::extension::VariableDoc(
  "size_f",
  "(float, float)",
  "The size of the bounding box as float values, read access only"
);
PyObject* PyBobIpFacedetectBoundingBox_size_f(PyBobIpFacedetectBoundingBoxObject* self, void*){
  BOB_TRY
  return Py_BuildValue("dd", self->cxx->height(), self->cxx->width());
  BOB_CATCH_MEMBER("size_f could not be read", 0)
}


static auto center = bob::extension::VariableDoc(
  "center",
  "(float, float)",
  "The center of the bounding box (as float values), read access only"
);
PyObject* PyBobIpFacedetectBoundingBox_center(PyBobIpFacedetectBoundingBoxObject* self, void*){
  BOB_TRY
  return Py_BuildValue("dd", self->cxx->center()[0], self->cxx->center()[1]);
  BOB_CATCH_MEMBER("center could not be read", 0)
}

static auto area = bob::extension::VariableDoc(
  "area",
  "float",
  "The area (height x width) of the bounding box, read access only"
);
PyObject* PyBobIpFacedetectBoundingBox_area(PyBobIpFacedetectBoundingBoxObject* self, void*){
  BOB_TRY
  return Py_BuildValue("d", self->cxx->area());
  BOB_CATCH_MEMBER("area could not be read", 0)
}

static PyGetSetDef PyBobIpFacedetectBoundingBox_getseters[] = {
    {
      topleft.name(),
      (getter)PyBobIpFacedetectBoundingBox_topleft,
      0,
      topleft.doc(),
      0
    },
    {
      top.name(),
      (getter)PyBobIpFacedetectBoundingBox_top,
      0,
      top.doc(),
      0
    },
    {
      left.name(),
      (getter)PyBobIpFacedetectBoundingBox_left,
      0,
      left.doc(),
      0
    },
    {
      topleft_f.name(),
      (getter)PyBobIpFacedetectBoundingBox_topleft_f,
      0,
      topleft_f.doc(),
      0
    },
    {
      top_f.name(),
      (getter)PyBobIpFacedetectBoundingBox_top_f,
      0,
      top_f.doc(),
      0
    },
    {
      left_f.name(),
      (getter)PyBobIpFacedetectBoundingBox_left_f,
      0,
      left_f.doc(),
      0
    },
    {
      bottomright.name(),
      (getter)PyBobIpFacedetectBoundingBox_bottomright,
      0,
      bottomright.doc(),
      0
    },
    {
      bottom.name(),
      (getter)PyBobIpFacedetectBoundingBox_bottom,
      0,
      bottom.doc(),
      0
    },
    {
      right.name(),
      (getter)PyBobIpFacedetectBoundingBox_right,
      0,
      right.doc(),
      0
    },
    {
      bottomright_f.name(),
      (getter)PyBobIpFacedetectBoundingBox_bottomright_f,
      0,
      bottomright_f.doc(),
      0
    },
    {
      bottom_f.name(),
      (getter)PyBobIpFacedetectBoundingBox_bottom_f,
      0,
      bottom_f.doc(),
      0
    },
    {
      right_f.name(),
      (getter)PyBobIpFacedetectBoundingBox_right_f,
      0,
      right_f.doc(),
      0
    },
    {
      size.name(),
      (getter)PyBobIpFacedetectBoundingBox_size,
      0,
      size.doc(),
      0
    },
    {
      size_f.name(),
      (getter)PyBobIpFacedetectBoundingBox_size_f,
      0,
      size_f.doc(),
      0
    },
    {
      center.name(),
      (getter)PyBobIpFacedetectBoundingBox_center,
      0,
      center.doc(),
      0
    },
    {
      area.name(),
      (getter)PyBobIpFacedetectBoundingBox_area,
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

static PyObject* PyBobIpFacedetectBoundingBox_shift(PyBobIpFacedetectBoundingBoxObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY
  char** kwlist = shift.kwlist();

  blitz::TinyVector<double,2> offset;
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "(dd)", kwlist, &offset[0], &offset[1])) return 0;
  PyBobIpFacedetectBoundingBoxObject* ret = reinterpret_cast<PyBobIpFacedetectBoundingBoxObject*>(PyBobIpFacedetectBoundingBox_Type.tp_alloc(&PyBobIpFacedetectBoundingBox_Type, 0));
  ret->cxx = self->cxx->shift(offset[0], offset[1]);

  return Py_BuildValue("N", ret);
  BOB_CATCH_MEMBER("cannot shift", 0)
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

static PyObject* PyBobIpFacedetectBoundingBox_scale(PyBobIpFacedetectBoundingBoxObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY
  char** kwlist = scale.kwlist();

  double scale;
  PyObject* centered = 0;
  // by shape
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "d|O!", kwlist, &scale, &PyBool_Type, &centered)){
    return 0;
  }
  PyBobIpFacedetectBoundingBoxObject* ret = reinterpret_cast<PyBobIpFacedetectBoundingBoxObject*>(PyBobIpFacedetectBoundingBox_Type.tp_alloc(&PyBobIpFacedetectBoundingBox_Type, 0));
  if (f(centered))
    ret->cxx = self->cxx->scaleCentered(scale);
  else
    ret->cxx = self->cxx->scale(scale);
  return Py_BuildValue("N", ret);
  BOB_CATCH_MEMBER("cannot scale", 0)
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
static PyObject* PyBobIpFacedetectBoundingBox_mirror_x(PyBobIpFacedetectBoundingBoxObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY
  char** kwlist = mirror_x.kwlist();

  int width;
  // by shape
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "i", kwlist, &width)){
    return 0;
  }
  PyBobIpFacedetectBoundingBoxObject* ret = reinterpret_cast<PyBobIpFacedetectBoundingBoxObject*>(PyBobIpFacedetectBoundingBox_Type.tp_alloc(&PyBobIpFacedetectBoundingBox_Type, 0));
  ret->cxx = self->cxx->mirrorX(width);
  return Py_BuildValue("N", ret);
  BOB_CATCH_MEMBER("cannot mirror horizontally", 0)
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
static PyObject* PyBobIpFacedetectBoundingBox_overlap(PyBobIpFacedetectBoundingBoxObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY
  char** kwlist = overlap.kwlist();

  PyBobIpFacedetectBoundingBoxObject* other;
  // by shape
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!", kwlist, &PyBobIpFacedetectBoundingBox_Type, &other)) return 0;

  PyBobIpFacedetectBoundingBoxObject* ret = reinterpret_cast<PyBobIpFacedetectBoundingBoxObject*>(PyBobIpFacedetectBoundingBox_Type.tp_alloc(&PyBobIpFacedetectBoundingBox_Type, 0));
  ret->cxx = self->cxx->overlap(*other->cxx);
  return Py_BuildValue("N", ret);
  BOB_CATCH_MEMBER("cannot compute overlap", 0)
}

static auto similarity = bob::extension::FunctionDoc(
  "similarity",
  "This function computes the Jaccard similarity index between this and the given BoundingBox",
  "The Jaccard similarity coefficient between two bounding boxes is defined as their intersection divided by their union:\n\n"
  ".. math::\n\n   J(A,B) = \\frac{|A\\cap B|}{|A\\cup B|}",
  true
)
.add_prototype("other", "sim")
.add_parameter("other", ":py:class:`BoundingBox`", "The other bounding box to compute the overlap with")
.add_return("sim", "float", "The Jaccard similarity index between this and the given BoundingBox")
;
static PyObject* PyBobIpFacedetectBoundingBox_similarity(PyBobIpFacedetectBoundingBoxObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY
  char** kwlist = similarity.kwlist();

  PyBobIpFacedetectBoundingBoxObject* other;
  // by shape
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!", kwlist, &PyBobIpFacedetectBoundingBox_Type, &other)){
    return 0;
  }

  double sim = self->cxx->similarity(*other->cxx);
  return Py_BuildValue("d", sim);
  BOB_CATCH_MEMBER("cannot compute overlap", 0)
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
static PyObject* PyBobIpFacedetectBoundingBox_is_valid_for(PyBobIpFacedetectBoundingBoxObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY
  char** kwlist = is_valid_for.kwlist();

  blitz::TinyVector<int,2> size;
  // by shape
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "(ii)", kwlist, &size[0], &size[1])){
    return 0;
  }
  if (self->cxx->isValidFor(size))
    Py_RETURN_TRUE;
  else
    Py_RETURN_FALSE;
  BOB_CATCH_MEMBER("cannot compute validity", 0)
}


static auto contains = bob::extension::FunctionDoc(
  "contains",
  "Checks if the bounding box contains the given point",
  0,
  true
)
.add_prototype("point", "contained")
.add_parameter("point", "(float, float)", "The point to test")
.add_return("contained", "bool", "``True`` if the bounding box contains the given point, ``False`` otherwise")
;
static PyObject* PyBobIpFacedetectBoundingBox_contains(PyBobIpFacedetectBoundingBoxObject* self, PyObject* args, PyObject* kwargs) {
  BOB_TRY
  char** kwlist = contains.kwlist();

  blitz::TinyVector<double,2> point;
  // by shape
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "(dd)", kwlist, &point[0], &point[1])){
    return 0;
  }
  if (self->cxx->contains(point))
    Py_RETURN_TRUE;
  else
    Py_RETURN_FALSE;
  BOB_CATCH_MEMBER("cannot compute point", 0)
}


static PyMethodDef PyBobIpFacedetectBoundingBox_methods[] = {
  {
    scale.name(),
    (PyCFunction)PyBobIpFacedetectBoundingBox_scale,
    METH_VARARGS|METH_KEYWORDS,
    scale.doc()
  },
  {
    shift.name(),
    (PyCFunction)PyBobIpFacedetectBoundingBox_shift,
    METH_VARARGS|METH_KEYWORDS,
    shift.doc()
  },
  {
    mirror_x.name(),
    (PyCFunction)PyBobIpFacedetectBoundingBox_mirror_x,
    METH_VARARGS|METH_KEYWORDS,
    mirror_x.doc()
  },
  {
    overlap.name(),
    (PyCFunction)PyBobIpFacedetectBoundingBox_overlap,
    METH_VARARGS|METH_KEYWORDS,
    overlap.doc()
  },
  {
    similarity.name(),
    (PyCFunction)PyBobIpFacedetectBoundingBox_similarity,
    METH_VARARGS|METH_KEYWORDS,
    similarity.doc()
  },
  {
    is_valid_for.name(),
    (PyCFunction)PyBobIpFacedetectBoundingBox_is_valid_for,
    METH_VARARGS|METH_KEYWORDS,
    is_valid_for.doc()
  },
  {
    contains.name(),
    (PyCFunction)PyBobIpFacedetectBoundingBox_contains,
    METH_VARARGS|METH_KEYWORDS,
    contains.doc()
  },
  {0} /* Sentinel */
};


/******************************************************************/
/************ Module Section **************************************/
/******************************************************************/

// Define the DCTFeatures type struct; will be initialized later
PyTypeObject PyBobIpFacedetectBoundingBox_Type = {
  PyVarObject_HEAD_INIT(0,0)
  0
};

bool init_BobIpFacedetectBoundingBox(PyObject* module)
{
  // initialize the type struct
  PyBobIpFacedetectBoundingBox_Type.tp_name = BoundingBox_doc.name();
  PyBobIpFacedetectBoundingBox_Type.tp_basicsize = sizeof(PyBobIpFacedetectBoundingBoxObject);
  PyBobIpFacedetectBoundingBox_Type.tp_flags = Py_TPFLAGS_DEFAULT;
  PyBobIpFacedetectBoundingBox_Type.tp_doc = BoundingBox_doc.doc();
  PyBobIpFacedetectBoundingBox_Type.tp_repr = (reprfunc)PyBobIpFacedetectBoundingBox_Str;
  PyBobIpFacedetectBoundingBox_Type.tp_str = (reprfunc)PyBobIpFacedetectBoundingBox_Str;

  // set the functions
  PyBobIpFacedetectBoundingBox_Type.tp_new = PyType_GenericNew;
  PyBobIpFacedetectBoundingBox_Type.tp_init = reinterpret_cast<initproc>(PyBobIpFacedetectBoundingBox_init);
  PyBobIpFacedetectBoundingBox_Type.tp_dealloc = reinterpret_cast<destructor>(PyBobIpFacedetectBoundingBox_delete);
  PyBobIpFacedetectBoundingBox_Type.tp_richcompare = reinterpret_cast<richcmpfunc>(PyBobIpFacedetectBoundingBox_RichCompare);
  PyBobIpFacedetectBoundingBox_Type.tp_methods = PyBobIpFacedetectBoundingBox_methods;
  PyBobIpFacedetectBoundingBox_Type.tp_getset = PyBobIpFacedetectBoundingBox_getseters;

  // check that everything is fine
  if (PyType_Ready(&PyBobIpFacedetectBoundingBox_Type) < 0) return false;

  // add the type to the module
  Py_INCREF(&PyBobIpFacedetectBoundingBox_Type);
  return PyModule_AddObject(module, "BoundingBox", (PyObject*)&PyBobIpFacedetectBoundingBox_Type) >= 0;
}
