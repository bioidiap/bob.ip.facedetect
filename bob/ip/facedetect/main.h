/**
 * @author Manuel Guenther <manuel.guenther@idiap.ch>
 * @date Mon Jun 23 19:18:25 CEST 2014
 *
 * @brief Header file for bindings to bob::ip
 */

#ifndef FACERECLIB_FACEDETECT_MAIN_H
#define FACERECLIB_FACEDETECT_MAIN_H

#include <bob.blitz/cppapi.h>
#include <bob.blitz/cleanup.h>
#include <bob.core/api.h>
#include <bob.io.base/api.h>
#include <bob.ip.base/api.h>
#include <bob.extension/documentation.h>

#include "cpp/features.h"

static inline bool f(PyObject* o){return o != 0 && PyObject_IsTrue(o) > 0;}  /* converts PyObject to bool and returns false if object is NULL */

// BoundingBox
typedef struct {
  PyObject_HEAD
  boost::shared_ptr<bob::ip::facedetect::BoundingBox> cxx;
} PyBobIpFacedetectBoundingBoxObject;

extern PyTypeObject PyBobIpFacedetectBoundingBox_Type;
bool init_BobIpFacedetectBoundingBox(PyObject* module);
int PyBobIpFacedetectBoundingBox_Check(PyObject* o);

// Feature extractor
typedef struct {
  PyObject_HEAD
  boost::shared_ptr<bob::ip::facedetect::FeatureExtractor> cxx;
} PyBobIpFacedetectFeatureExtractorObject;

extern PyTypeObject PyBobIpFacedetectFeatureExtractor_Type;
bool init_BobIpFacedetectFeatureExtractor(PyObject* module);
int PyBobIpFacedetectFeatureExtractor_Check(PyObject* o);

// Functions
PyObject* PyBobIpFacedetect_PruneDetections(PyObject*, PyObject*, PyObject*);
extern bob::extension::FunctionDoc prune_detections_doc;
PyObject* PyBobIpFacedetect_OverlappingDetections(PyObject*, PyObject*, PyObject*);
extern bob::extension::FunctionDoc overlapping_detections_doc;

#endif // FACERECLIB_FACEDETECT_MAIN_H
