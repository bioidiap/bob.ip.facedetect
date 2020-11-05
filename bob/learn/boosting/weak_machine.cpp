
#include "main.h"
#include <boost/type_index.hpp>
#include <boost/functional/hash.hpp>

static std::map<size_t,CreateFunction> machineFactory;

bool registerMachineType(size_t type_hash, CreateFunction creator_function){
  if (machineFactory.find(type_hash) != machineFactory.end()){
    PyErr_Format(PyExc_TypeError, "The given machine hash %zu already has been registered.", type_hash);
    return false;
  }
  machineFactory[type_hash] = creator_function;
  return true;
}

PyObject* createMachine(boost::shared_ptr<bob::learn::boosting::WeakMachine> machine){
  size_t type_hash = boost::typeindex::type_id_runtime(*machine).hash_code();
  if (machineFactory.find(type_hash) == machineFactory.end()){
    PyErr_Format(PyExc_TypeError, "The given machine hash %zu has not been registered.", type_hash);
    return NULL;
  }
  return machineFactory[type_hash](machine);
}

static PyObject* weakMachineCreate(boost::shared_ptr<bob::learn::boosting::WeakMachine> machine){
  PyObject* o = WeakMachineType.tp_alloc(&WeakMachineType,0);
  reinterpret_cast<WeakMachineObject*>(o)->base = machine;
  return o;
}

int weakMachineCheck(PyObject* o){
  if (!o) return 0;
  return PyObject_IsInstance(o, reinterpret_cast<PyObject*>(&WeakMachineType));
}

int weakMachineConverter(PyObject* o, WeakMachineObject** a) {
  if (!weakMachineCheck(o)) return 0;
  Py_INCREF(o);
  (*a) = reinterpret_cast<WeakMachineObject*>(o);
  return 1;
}


static auto weakMachine_doc = bob::extension::ClassDoc(
  "WeakMachine",
  "Pure virtual base class for weak machines"
);

// Define Weak Machine type here to avoid needing another cpp file
PyTypeObject WeakMachineType = {
  PyVarObject_HEAD_INIT(0,0)
  0
};

bool init_WeakMachine(PyObject* module){
  // The weak machine is quite simple since it is pure virtual
  WeakMachineType.tp_name = weakMachine_doc.name();
  WeakMachineType.tp_basicsize = sizeof(WeakMachineObject);
  WeakMachineType.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
  WeakMachineType.tp_doc = weakMachine_doc.doc();

  // register creator function
  // register machine
  if (!registerMachineType(boost::typeindex::type_id<bob::learn::boosting::WeakMachine>().hash_code(), &weakMachineCreate))
    return false;

  // check that everyting is fine
  if (PyType_Ready(&WeakMachineType) < 0)
    return false;

  // add the type to the module
  Py_INCREF(&WeakMachineType);
  return PyModule_AddObject(module, weakMachine_doc.name(), (PyObject*)&WeakMachineType) >= 0;
}

