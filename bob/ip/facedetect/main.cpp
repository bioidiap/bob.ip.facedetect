/**
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 * @date Mon Apr 14 20:45:21 CEST 2014
 *
 * @brief Bindings to bob::ip routines
 */

#ifdef NO_IMPORT_ARRAY
#undef NO_IMPORT_ARRAY
#endif
#include "main.h"

bob::extension::FunctionDoc prune_detections_doc = bob::extension::FunctionDoc(
  "prune_detections",
  "Prunes the given detected bounding boxes according to their predictions and returns the pruned bounding boxes and their predictions",
  "For threshold >= 1., all detections will be returned (i.e., no pruning is performed), but the list will be sorted with descendant predictions."
)
.add_prototype("detections, predictions, threshold, [number_of_detections]", "pruned_detections, pruned_predictions")
.add_parameter("detections", "[:py:class:`BoundingBox`]", "A list of detected bounding boxes")
.add_parameter("predictions", "array_like <1D, float>", "The prediction (quality, weight, ...) values for the detections")
.add_parameter("threshold", "float", "The overlap threshold (Jaccard similarity), for which detections should be pruned")
.add_parameter("number_of_detections", "int", "[default: MAX_INT] The number of detections that should be returned")
.add_return("pruned_detections", "[:py:class:`BoundingBox`]", "The list of pruned bounding boxes")
.add_return("pruned_predictions", "array_like <float, 1D>", "The according predictions (qualities, weights, ...)")
;
PyObject* PyBobIpFacedetect_PruneDetections(PyObject*, PyObject* args, PyObject* kwargs) {
  BOB_TRY
  char** kwlist = prune_detections_doc.kwlist();

  PyObject* list;
  PyBlitzArrayObject* predictions;
  double threshold;
  int number_of_detections = std::numeric_limits<int>::max();

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!O&d|i", kwlist, &PyList_Type, &list, &PyBlitzArray_Converter, &predictions, &threshold, &number_of_detections)) return 0;
  auto predictions_ = make_safe(predictions);
  auto p = PyBlitzArrayCxx_AsBlitz<double,1>(predictions, "predictions");
  if (!p) return 0;

  // get bounding box list
  std::vector<boost::shared_ptr<bob::ip::facedetect::BoundingBox>> boxes(PyList_GET_SIZE(list)), pruned_boxes;
  for (Py_ssize_t i = 0; i < PyList_GET_SIZE(list); ++i){
    PyObject* v = PyList_GET_ITEM(list, i);
    if (!PyBobIpFacedetectBoundingBox_Check(v)){
      PyErr_Format(PyExc_TypeError, "prune_detections : expected a list of BoundingBox objects, but object number %d is of type `%s'", (int)i, Py_TYPE(v)->tp_name);
      return 0;
    }
    boxes[i] = ((PyBobIpFacedetectBoundingBoxObject*)v)->cxx;
  }

  blitz::Array<double,1> pruned_predictions;

  // perform pruning
  bob::ip::facedetect::pruneDetections(boxes, *p, threshold, pruned_boxes, pruned_predictions, number_of_detections);

  // re-transform boxes into python list
  PyObject* pruned = PyList_New(pruned_boxes.size());
  for (Py_ssize_t i = 0; i < PyList_GET_SIZE(pruned); ++i){
    PyBobIpFacedetectBoundingBoxObject* bb = reinterpret_cast<PyBobIpFacedetectBoundingBoxObject*>(PyBobIpFacedetectBoundingBox_Type.tp_alloc(&PyBobIpFacedetectBoundingBox_Type, 0));
    bb->cxx = pruned_boxes[i];
    PyList_SET_ITEM(pruned, i, Py_BuildValue("N", bb));
  }

  // return tuple: detections, predictions
  return Py_BuildValue("NN", pruned, PyBlitzArrayCxx_AsNumpy(pruned_predictions));

  BOB_CATCH_FUNCTION("in prune_detections", 0)
}

bob::extension::FunctionDoc group_detections_doc = bob::extension::FunctionDoc(
  "group_detections",
  "Groups the given detected bounding boxes according to their overlap and returns a list of lists of detections, and their according list of predictions",
  "Each of the returned lists of bounding boxes contains all boxes that overlap with the first box in the list with at least the given ``overlap_threshold``."
)
.add_prototype("detections, predictions, overlap_threshold, prediction_threshold, box_count_threshold", "grouped_detections, grouped_predictions")
.add_parameter("detections", "[:py:class:`BoundingBox`]", "A list of detected bounding boxes")
.add_parameter("predictions", "array_like <1D, float>", "The prediction (quality, weight, ...) values for the detections")
.add_parameter("overlap_threshold", "float", "The overlap threshold (Jaccard similarity), for which detections should be considered to overlap")
.add_parameter("prediction_threshold", "float", "[Default: ``0``] The prediction threshold, below which the bounding boxes should be disregarded and not added to any group")
.add_parameter("box_count_threshold", "int", "[Default: ``1``] Only bounding boxes with at least the given number of overlapping boxes are considered")
.add_return("grouped_detections", "[[:py:class:`BoundingBox`]]", "The lists of bounding boxes that are grouped by their overlap; each list contains all bounding boxes that overlap with the first entry in the list")
.add_return("grouped_predictions", "[array_like <float, 1D>]", "The according list of grouped predictions (qualities, weights, ...)")
;
PyObject* PyBobIpFacedetect_GroupDetections(PyObject*, PyObject* args, PyObject* kwargs) {
  BOB_TRY
  char** kwlist = group_detections_doc.kwlist();

  PyObject* list;
  PyBlitzArrayObject* predictions;
  double overlap_threshold, prediction_threshold=0.;
  int box_count_threshold = 1;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!O&d|di", kwlist, &PyList_Type, &list, &PyBlitzArray_Converter, &predictions, &overlap_threshold, &prediction_threshold, &box_count_threshold)) return 0;
  auto predictions_ = make_safe(predictions);
  auto p = PyBlitzArrayCxx_AsBlitz<double,1>(predictions, "predictions");
  if (!p) return 0;

  // get bounding box list
  std::vector<boost::shared_ptr<bob::ip::facedetect::BoundingBox>> boxes(PyList_GET_SIZE(list));
  for (Py_ssize_t i = 0; i < PyList_GET_SIZE(list); ++i){
    PyObject* v = PyList_GET_ITEM(list, i);
    if (!PyBobIpFacedetectBoundingBox_Check(v)){
      PyErr_Format(PyExc_TypeError, "prune_detections : expected a list of BoundingBox objects, but object number %d is of type `%s'", (int)i, Py_TYPE(v)->tp_name);
      return 0;
    }
    boxes[i] = ((PyBobIpFacedetectBoundingBoxObject*)v)->cxx;
  }

  std::vector<blitz::Array<double,1>> grouped_predictions;
  std::vector<std::vector<boost::shared_ptr<bob::ip::facedetect::BoundingBox>>> grouped_boxes;

  // perform pruning
  bob::ip::facedetect::groupDetections(boxes, *p, overlap_threshold, prediction_threshold, box_count_threshold, grouped_boxes, grouped_predictions);

  // re-transform boxes and predictions into python list
  PyObject* all_boxes = PyList_New(grouped_boxes.size());
  PyObject* all_predictions = PyList_New(grouped_predictions.size());
  for (Py_ssize_t i = 0; i < PyList_GET_SIZE(all_boxes); ++i){
    // set predictions
    PyList_SET_ITEM(all_predictions, i, PyBlitzArrayCxx_AsNumpy(grouped_predictions[i]));

    // create list of bounding boxes and add it to the list
    PyObject* boxes = PyList_New(grouped_boxes[i].size());
    PyList_SET_ITEM(all_boxes, i, boxes);
    // now, fill the list
    for (Py_ssize_t j = 0; j < PyList_GET_SIZE(boxes); ++j){
      PyBobIpFacedetectBoundingBoxObject* bb = reinterpret_cast<PyBobIpFacedetectBoundingBoxObject*>(PyBobIpFacedetectBoundingBox_Type.tp_alloc(&PyBobIpFacedetectBoundingBox_Type, 0));
      bb->cxx = grouped_boxes[i][j];
      PyList_SET_ITEM(boxes, j, Py_BuildValue("N", bb));
    }
  }

  // return tuple: detections, predictions
  return Py_BuildValue("NN", all_boxes, all_predictions);

  BOB_CATCH_FUNCTION("group_detections", 0)
}


bob::extension::FunctionDoc overlapping_detections_doc = bob::extension::FunctionDoc(
  "overlapping_detections",
  "Returns the detections and predictions that overlap with the best detection",
  "For threshold >= 1., all detections will be returned (i.e., no pruning is performed), but the list will be sorted with descendingly predictions."
)
.add_prototype("detections, predictions, threshold", "overlapped_detections, overlapped_predictions")
.add_parameter("detections", "[:py:class:`BoundingBox`]", "A list of detected bouding boxes")
.add_parameter("predictions", "array_like <1D, float>", "The prediction (quality, weight, ...) values for the detections")
.add_parameter("threshold", "float", "The overlap threshold (Jaccard similarity) which should be considered")
.add_return("overlapped_detections", "[:py:class:`BoundingBox`]", "The list of overlapping bounding boxes")
.add_return("overlapped_predictions", "array_like <float, 1D>", "The according predictions (qualities, weights, ...)")
;
PyObject* PyBobIpFacedetect_OverlappingDetections(PyObject*, PyObject* args, PyObject* kwargs) {
  BOB_TRY
  char** kwlist = overlapping_detections_doc.kwlist();

  PyObject* list;
  PyBlitzArrayObject* predictions;
  double threshold;

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!O&d", kwlist, &PyList_Type, &list, &PyBlitzArray_Converter, &predictions, &threshold)) return 0;
  auto predictions_ = make_safe(predictions);
  auto p = PyBlitzArrayCxx_AsBlitz<double,1>(predictions, "predictions");
  if (!p) return 0;

  // get bounding box list
  std::vector<boost::shared_ptr<bob::ip::facedetect::BoundingBox>> boxes(PyList_GET_SIZE(list)), overlapped_boxes;
  for (Py_ssize_t i = 0; i < PyList_GET_SIZE(list); ++i){
    PyObject* v = PyList_GET_ITEM(list, i);
    if (!PyBobIpFacedetectBoundingBox_Check(v)){
      PyErr_Format(PyExc_TypeError, "overlapping_detections : expected a list of BoundingBox objects, but object number %d is of type `%s'", (int)i, Py_TYPE(v)->tp_name);
      return 0;
    }
    boxes[i] = ((PyBobIpFacedetectBoundingBoxObject*)v)->cxx;
  }

  blitz::Array<double,1> overlapped_predictions;

  // perform pruning
  bob::ip::facedetect::bestOverlap(boxes, *p, threshold, overlapped_boxes, overlapped_predictions);

  // re-transform boxes into python list
  PyObject* overlapped = PyList_New(overlapped_boxes.size());
  for (Py_ssize_t i = 0; i < PyList_GET_SIZE(overlapped); ++i){
    PyBobIpFacedetectBoundingBoxObject* bb = reinterpret_cast<PyBobIpFacedetectBoundingBoxObject*>(PyBobIpFacedetectBoundingBox_Type.tp_alloc(&PyBobIpFacedetectBoundingBox_Type, 0));
    bb->cxx = overlapped_boxes[i];
    PyList_SET_ITEM(overlapped, i, Py_BuildValue("N", bb));
  }

  // return tuple: detections, predictions
  return Py_BuildValue("NN", overlapped, PyBlitzArrayCxx_AsNumpy(overlapped_predictions));

  BOB_CATCH_FUNCTION("in overlapping_detections", 0)
}


static PyMethodDef module_methods[] = {
  {
    prune_detections_doc.name(),
    (PyCFunction)PyBobIpFacedetect_PruneDetections,
    METH_VARARGS|METH_KEYWORDS,
    prune_detections_doc.doc()
  },
  {
    group_detections_doc.name(),
    (PyCFunction)PyBobIpFacedetect_GroupDetections,
    METH_VARARGS|METH_KEYWORDS,
    group_detections_doc.doc()
  },
  {
    overlapping_detections_doc.name(),
    (PyCFunction)PyBobIpFacedetect_OverlappingDetections,
    METH_VARARGS|METH_KEYWORDS,
    overlapping_detections_doc.doc()
  },
  {0}  // Sentinel
};


PyDoc_STRVAR(module_docstr, "C++ implementattions for face detection utilities");

#if PY_VERSION_HEX >= 0x03000000
static PyModuleDef module_definition = {
  PyModuleDef_HEAD_INIT,
  BOB_EXT_MODULE_NAME,
  module_docstr,
  -1,
  module_methods,
  0, 0, 0, 0
};
#endif

static PyObject* create_module (void) {

# if PY_VERSION_HEX >= 0x03000000
  PyObject* module = PyModule_Create(&module_definition);
  auto module_ = make_xsafe(module);
  const char* ret = "O";
# else
  PyObject* module = Py_InitModule3(BOB_EXT_MODULE_NAME, module_methods, module_docstr);
  const char* ret = "N";
# endif
  if (!module) return 0;

  if (!init_BobIpFacedetectBoundingBox(module)) return 0;
  if (!init_BobIpFacedetectFeatureExtractor(module)) return 0;

  /* imports bob.blitz C-API + dependencies */
  if (import_bob_blitz() < 0) return 0;
  if (import_bob_core_logging() < 0) return 0;
  if (import_bob_io_base() < 0) return 0;
  if (import_bob_ip_base() < 0) return 0;

  return Py_BuildValue(ret, module);
}

PyMODINIT_FUNC BOB_EXT_ENTRY_NAME (void) {
# if PY_VERSION_HEX >= 0x03000000
  return
# endif
    create_module();
}
