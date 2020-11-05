/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Thu 20 Sep 2012 14:46:35 CEST
 *
 * @brief Bob/Python extension to flandmark
 */

#include <bob.blitz/cppapi.h>
#include <bob.blitz/cleanup.h>
#include <bob.io.base/api.h>
#include <structmember.h>

#include <bob.extension/documentation.h>

#include <boost/shared_ptr.hpp>
#include <boost/shared_array.hpp>

#include <cstring>

#include "cpp/flandmark_detector.h"

/******************************************
 * Implementation of Localizer base class *
 ******************************************/

#define CLASS_NAME "Flandmark"

static auto s_class = bob::extension::ClassDoc(
  BOB_EXT_MODULE_PREFIX "." CLASS_NAME,

  "A key-point localization for faces using Flandmark",

  "This class can be used to locate facial landmarks on pre-detected faces. "
  "You input an image and a bounding-box specification and it returns you the "
  "positions for multiple key-points for the given face image.\n"
  "\n"
  "Consult http://cmp.felk.cvut.cz/~uricamic/flandmark/index.php for more "
  "information.\n"
  "\n"
)
.add_constructor(
  bob::extension::FunctionDoc(
    CLASS_NAME,
    "Constructor",
    "Initializes the key-point locator with a model."
    )
  .add_prototype("[model]", "")
  .add_parameter("model", "str (path), optional", "Path to the localization model. If not set (or set to ``None``), then use the default localization model, stored on the class variable ``__default_model__``)")
);

typedef struct {
  PyObject_HEAD
  bob::ip::flandmark::FLANDMARK_Model* flandmark;
  char* filename;
} PyBobIpFlandmarkObject;

static int PyBobIpFlandmark_init(PyBobIpFlandmarkObject* self, PyObject* args, PyObject* kwds) {
BOB_TRY
  /* Parses input arguments in a single shot */
  char** kwlist = s_class.kwlist();

  const char* model = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "|s", kwlist, &model)) return -1;

  if (!model) {
    //use what is stored in __default_model__
    PyObject* default_model = PyObject_GetAttrString((PyObject*)self,  "_default_model");
    if (!default_model) {
      PyErr_Format(PyExc_RuntimeError, "`%s' needs a model to properly initialize, but the user has not passed one and `_default_model' is not properly set", Py_TYPE(self)->tp_name);
      return -1;
    }
    auto default_model_ = make_safe(default_model);

    model = PyString_AS_STRING(default_model);
  }

  self->flandmark = bob::ip::flandmark::flandmark_init(model);
  if (!self->flandmark){
    PyErr_Format(PyExc_RuntimeError, "`%s' could not initialize from model file `%s'", Py_TYPE(self)->tp_name, model);
    return -1;
  }

  //flandmark is now initialized, set filename
  self->filename = strndup(model, 256);

  //all good, flandmark is ready
  return 0;
BOB_CATCH_MEMBER("constructor", -1)
}

static void PyBobIpFlandmark_delete (PyBobIpFlandmarkObject* self) {
  flandmark_free(self->flandmark);
  self->flandmark = 0;
  free(self->filename);
  self->filename = 0;
  Py_TYPE(self)->tp_free((PyObject*)self);
}


static auto s_locate = bob::extension::FunctionDoc(
  "locate",
  "Locates keypoints on a **single** facial bounding-box on the provided image.",
  "This method will locate 8 keypoints inside the bounding-box defined for the current input image, organized in this way:\n"
  "\n"
  "0. Face center\n"
  "1. Canthus-rl (inner corner of the right eye).\n"
  "\n"
  "   .. note::\n"
  "      \n"
  "      The \"right eye\" means the right eye at the face w.r.t. the person on the image. "
  "      That is the left eye in the image, from the viewer's perspective.\n"
  "\n"
  "2. Canthus-lr (inner corner of the left eye)\n"
  "3. Mouth-corner-r (right corner of the mouth)\n"
  "4. Mouth-corner-l (left corner of the mouth)\n"
  "5. Canthus-rr (outer corner of the right eye)\n"
  "6. Canthus-ll (outer corner of the left eye)\n"
  "7. Nose\n"
  "\n"
  "Each point is returned as tuple defining the pixel positions in the form ``(y, x)``.",
  true
)
.add_prototype("image, y, x, height, width", "landmarks")
.add_parameter("image", "array-like (2D, uint8)", "The image Flandmark will operate on")
.add_parameter("y, x", "int", "The top left-most corner of the bounding box containing the face image you want to locate keypoints on, defaults to 0.")
.add_parameter("height, width", "int", "The dimensions accross ``y`` (height) and ``x`` (width) for the bounding box, in number of pixels; defaults to full image resolution.")
.add_return("landmarks", "array (2D, float64)", "Each row in the output array contains the locations of keypoints in the format ``(y, x)``")
;

static PyObject* PyBobIpFlandmark_locate(PyBobIpFlandmarkObject* self,  PyObject *args, PyObject* kwds) {
BOB_TRY
  char** kwlist = s_locate.kwlist();

  PyBlitzArrayObject* image;
  int bbx[] = {0,0,-1,-1};

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&|iiii", kwlist,  &PyBlitzArray_Converter, &image, &bbx[0], &bbx[1], &bbx[2], &bbx[3])) return 0;


  auto image_ = make_safe(image);

  // check
  if (image->type_num != NPY_UINT8 || image->ndim != 2) {
    PyErr_Format(PyExc_TypeError, "`%s' input `image' data must be a 2D array with dtype `uint8' (i.e. a gray-scaled image), but you passed a %" PY_FORMAT_SIZE_T "d array with data type `%s'", Py_TYPE(self)->tp_name, image->ndim, PyBlitzArray_TypenumAsString(image->type_num));
    return 0;
  }

  auto cxx_image = PyBlitzArrayCxx_AsBlitz<uint8_t, 2>(image);
  // create bounding box in format (top, left, bottom, right)
  if (bbx[2] < 0)
    bbx[2] = cxx_image->extent(0);
  if (bbx[3] < 0)
    bbx[3] = cxx_image->extent(1);

  bbx[2] += bbx[0] - 1;
  bbx[3] += bbx[1] - 1;

  // detect
  std::vector<double> detected(2*self->flandmark->data.options.M);
  bob::ip::flandmark::flandmark_detect(*cxx_image, bbx, self->flandmark, &detected[0]);

  // extract landmarks
  blitz::Array<double, 2> landmarks(self->flandmark->data.options.M, 2);
  for (int k = 0; k < self->flandmark->data.options.M; ++k){
    landmarks(k,0) = detected[2*k];
    landmarks(k,1) = detected[2*k+1];
  }

  return PyBlitzArrayCxx_AsNumpy(landmarks);
BOB_CATCH_MEMBER("locate", 0)
};

static PyMethodDef PyBobIpFlandmark_methods[] = {
  {
    s_locate.name(),
    (PyCFunction)PyBobIpFlandmark_locate,
    METH_VARARGS|METH_KEYWORDS,
    s_locate.doc()
  },
  {0} /* Sentinel */
};

PyObject* PyBobIpFlandmark_Repr(PyBobIpFlandmarkObject* self) {

  /**
   * Expected output:
   *
   * <bob.ip.flandmark(model='...')>
   */

  PyObject* retval = PyUnicode_FromFormat("<%s(model='%s')>",  Py_TYPE(self)->tp_name, self->filename);

#if PYTHON_VERSION_HEX < 0x03000000
  if (!retval) return 0;
  PyObject* tmp = PyObject_Str(retval);
  Py_DECREF(retval);
  retval = tmp;
#endif

  return retval;

}

PyTypeObject PyBobIpFlandmark_Type = {
  PyVarObject_HEAD_INIT(0, 0)
  0
};

bool init_PyBobIpFlandmark(PyObject* module){
  // initialize the type struct
  PyBobIpFlandmark_Type.tp_name = s_class.name();
  PyBobIpFlandmark_Type.tp_basicsize = sizeof(PyBobIpFlandmarkObject);
  PyBobIpFlandmark_Type.tp_flags = Py_TPFLAGS_DEFAULT;
  PyBobIpFlandmark_Type.tp_doc = s_class.doc();
  PyBobIpFlandmark_Type.tp_dict = PyDict_New();

  // set the functions
  PyBobIpFlandmark_Type.tp_new = PyType_GenericNew;
  PyBobIpFlandmark_Type.tp_init = reinterpret_cast<initproc>(PyBobIpFlandmark_init);
  PyBobIpFlandmark_Type.tp_dealloc = reinterpret_cast<destructor>(PyBobIpFlandmark_delete);
  PyBobIpFlandmark_Type.tp_methods = PyBobIpFlandmark_methods;
  PyBobIpFlandmark_Type.tp_call = reinterpret_cast<ternaryfunc>(PyBobIpFlandmark_locate);
  PyBobIpFlandmark_Type.tp_str = reinterpret_cast<reprfunc>(PyBobIpFlandmark_Repr);
  PyBobIpFlandmark_Type.tp_repr = reinterpret_cast<reprfunc>(PyBobIpFlandmark_Repr);

  // check that everything is fine
  if (PyType_Ready(&PyBobIpFlandmark_Type) < 0) return false;

  // add the type to the module
  Py_INCREF(&PyBobIpFlandmark_Type);
  return PyModule_AddObject(module, "Flandmark", (PyObject*)&PyBobIpFlandmark_Type) >= 0;
}
