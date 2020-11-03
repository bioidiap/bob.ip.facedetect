#ifndef BOB_LEARN_BOOSTING_MAIN_H
#define BOB_LEARN_BOOSTING_MAIN_H

#include <bob.blitz/cppapi.h>
#include <bob.blitz/cleanup.h>
#include <bob.core/api.h>
#include <bob.io.base/api.h>
#include <bob.extension/documentation.h>

#include <boost/shared_ptr.hpp>

#include <bob.learn.boosting/LossFunction.h>
#include <bob.learn.boosting/JesorskyLoss.h>
#include <bob.learn.boosting/WeakMachine.h>
#include <bob.learn.boosting/StumpMachine.h>
#include <bob.learn.boosting/LUTMachine.h>
#include <bob.learn.boosting/BoostedMachine.h>
#include <bob.learn.boosting/LUTTrainer.h>

// helper function to convert const char* to char*
inline char* c(const char* o){return const_cast<char*>(o);}

// Loss function
typedef struct {
  PyObject_HEAD
  boost::shared_ptr<bob::learn::boosting::LossFunction> base;
} LossFunctionObject;

extern PyTypeObject LossFunctionType;

bool init_LossFunction(PyObject*);

// Jesorsky loss
typedef struct {
  LossFunctionObject parent;
  boost::shared_ptr<bob::learn::boosting::JesorskyLoss> base;
} JesorskyLossObject;

extern PyTypeObject JesorskyLossType;

bool init_JesorskyLoss(PyObject*);


// Weak machine
typedef PyObject*(*CreateFunction)(boost::shared_ptr<bob::learn::boosting::WeakMachine>);
bool registerMachineType(size_t, CreateFunction);
PyObject* createMachine(boost::shared_ptr<bob::learn::boosting::WeakMachine>);

typedef struct {
  PyObject_HEAD
  boost::shared_ptr<bob::learn::boosting::WeakMachine> base;
} WeakMachineObject;

extern PyTypeObject WeakMachineType;

int weakMachineCheck(PyObject*);

int weakMachineConverter(PyObject*, WeakMachineObject**);

bool init_WeakMachine(PyObject*);

// Stump machine
typedef struct {
  WeakMachineObject parent;
  boost::shared_ptr<bob::learn::boosting::StumpMachine> base;
} StumpMachineObject;

extern PyTypeObject StumpMachineType;

bool init_StumpMachine(PyObject*);

// LUT machine
typedef struct {
  WeakMachineObject parent;
  boost::shared_ptr<bob::learn::boosting::LUTMachine> base;
} LUTMachineObject;

extern PyTypeObject LUTMachineType;

bool init_LUTMachine(PyObject*);

// LUT machine
typedef struct {
  PyObject_HEAD
  boost::shared_ptr<bob::learn::boosting::BoostedMachine> base;
} BoostedMachineObject;

extern PyTypeObject BoostedMachineType;

bool init_BoostedMachine(PyObject*);


// LUT trainer
typedef struct {
  PyObject_HEAD
  boost::shared_ptr<bob::learn::boosting::LUTTrainer> base;
} LUTTrainerObject;

extern PyTypeObject LUTTrainerType;

bool init_LUTTrainer(PyObject*);


#endif // BOB_LEARN_BOOSTING_MAIN_H
