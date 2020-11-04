/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Fri 25 Oct 16:54:55 2013
 *
 * @brief Bindings to bob::ap
 */

#ifdef NO_IMPORT_ARRAY
#undef NO_IMPORT_ARRAY
#endif
#include <bob.blitz/capi.h>
#include <bob.blitz/cleanup.h>
#include <bob.core/api.h>
#include <bob.io.base/api.h>
#include <bob.extension/documentation.h>

extern PyTypeObject PyBobIpFlandmark_Type;
extern bool init_PyBobIpFlandmark(PyObject*);

static auto s_setter = bob::extension::FunctionDoc(
  "_set_default_model",
  "Internal function to set the default model for the Flandmark class"
)
.add_prototype("path", "")
.add_parameter("path", "str", "The path to the new model file")
;

PyObject* set_flandmark_model(PyObject*, PyObject* o) {
BOB_TRY
  int ok = PyDict_SetItemString(PyBobIpFlandmark_Type.tp_dict, "_default_model", o);

  if (ok == -1) return 0;

  Py_RETURN_NONE;
BOB_CATCH_FUNCTION("_set_default_model", 0)
}

static PyMethodDef module_methods[] = {
  {
    s_setter.name(),
    (PyCFunction)set_flandmark_model,
    METH_O,
    s_setter.doc()
  },
  {0}  /* Sentinel */
};

PyDoc_STRVAR(module_docstr, "Flandmark keypoint localization library");

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

  if (!init_PyBobIpFlandmark(module)) return 0;

  /* imports dependencies */
  if (import_bob_blitz() < 0) return 0;
  if (import_bob_core_logging() < 0) return 0;
  if (import_bob_io_base() < 0) return 0;

  return Py_BuildValue(ret, module);
}

PyMODINIT_FUNC BOB_EXT_ENTRY_NAME (void) {
# if PY_VERSION_HEX >= 0x03000000
  return
# endif
    create_module();
}
