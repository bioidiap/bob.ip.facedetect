/**
 * @author Manuel Guenther <manuel.guenther@idiap.ch>
 * @date Mon Jun 23 19:18:25 CEST 2014
 *
 * @brief Header file for bindings to bob::ip
 */

#ifndef FACERECLIB_FACEDETECT_MAIN_H
#define FACERECLIB_FACEDETECT_MAIN_H

#include <Python.h>

#include <bob.blitz/cppapi.h>
#include <bob.blitz/cleanup.h>
#include <bob.io.base/api.h>
#include <bob.sp/api.h>
#include <bob.ip.base/api.h>
#include <bob.extension/documentation.h>

#include "cpp/features.h"

#if PY_VERSION_HEX >= 0x03000000
#define PyInt_Check PyLong_Check
#define PyInt_AS_LONG PyLong_AS_LONG
#define PyString_Check PyUnicode_Check
#define PyString_AS_STRING(x) PyBytes_AS_STRING(make_safe(PyUnicode_AsUTF8String(x)).get())
#define PyString_FromString PyUnicode_FromString)
#endif

#define TRY try{

#define CATCH(message,ret) }\
  catch (std::exception& e) {\
    PyErr_SetString(PyExc_RuntimeError, e.what());\
    return ret;\
  } \
  catch (...) {\
    PyErr_Format(PyExc_RuntimeError, "%s " message ": unknown exception caught", Py_TYPE(self)->tp_name);\
    return ret;\
  }

#define CATCH_(message, ret) }\
  catch (std::exception& e) {\
    PyErr_SetString(PyExc_RuntimeError, e.what());\
    return ret;\
  } \
  catch (...) {\
    PyErr_Format(PyExc_RuntimeError, message ": unknown exception caught");\
    return ret;\
  }

static inline char* c(const char* o){return const_cast<char*>(o);}  /* converts const char* to char* */
static inline bool f(PyObject* o){return o != 0 && PyObject_IsTrue(o) > 0;}  /* converts PyObject to bool and returns false if object is NULL */

// BoundingBox
typedef struct {
  PyObject_HEAD
  boost::shared_ptr<BoundingBox> cxx;
} BoundingBoxObject;

extern PyTypeObject BoundingBox_Type;
bool init_BoundingBox(PyObject* module);
int BoundingBox_Check(PyObject* o);

// Feature extractor
typedef struct {
  PyObject_HEAD
  boost::shared_ptr<FeatureExtractor> cxx;
} FeatureExtractorObject;

extern PyTypeObject FeatureExtractor_Type;
bool init_FeatureExtractor(PyObject* module);
int FeatureExtractor_Check(PyObject* o);

// Functions
PyObject* prune_detections(PyObject*, PyObject*, PyObject*);
extern bob::extension::FunctionDoc prune_detections_doc;
PyObject* overlapping_detections(PyObject*, PyObject*, PyObject*);
extern bob::extension::FunctionDoc overlapping_detections_doc;

#endif // FACERECLIB_FACEDETECT_MAIN_H
