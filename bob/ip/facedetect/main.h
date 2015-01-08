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
