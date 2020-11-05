#ifndef BOB_LEARN_BOOSTING_WEAK_MACHINE_H
#define BOB_LEARN_BOOSTING_WEAK_MACHINE_H


#include <bob.io.base/HDF5File.h>

namespace bob { namespace learn { namespace boosting {

  /**
   * This is the pure virtual base class for all weak machines.
   */
  class WeakMachine{
    public:
      // uni-variate forwarding of a single feature
      virtual double forward(const blitz::Array<uint16_t, 1>& features) const {throw std::runtime_error("This function is not implemented for the given data type in the current class.");}
      virtual double forward(const blitz::Array<double, 1>& features) const {throw std::runtime_error("This function is not implemented for the given data type in the current class.");}

      // multi-variate forwarding of a single feature
      virtual void forward(const blitz::Array<uint16_t, 1>& features, blitz::Array<double,1> predictions) const {throw std::runtime_error("This function is not implemented for the given data type in the current class.");}
      virtual void forward(const blitz::Array<double, 1>& features, blitz::Array<double,1> predictions) const {throw std::runtime_error("This function is not implemented for the given data type in the current class.");}

      // uni-variate forwarding of a set of features
      virtual void forward(const blitz::Array<uint16_t, 2>& features, blitz::Array<double,1> predictions) const {throw std::runtime_error("This function is not implemented for the given data type in the current class.");}
      virtual void forward(const blitz::Array<double, 2>& features, blitz::Array<double,1> predictions) const {throw std::runtime_error("This function is not implemented for the given data type in the current class.");}

      // multi-variate forwarding of a set of features
      virtual void forward(const blitz::Array<uint16_t, 2>& features, blitz::Array<double,2> predictions) const {throw std::runtime_error("This function is not implemented for the given data type in the current class.");}
      virtual void forward(const blitz::Array<double, 2>& features, blitz::Array<double,2> predictions) const {throw std::runtime_error("This function is not implemented for the given data type in the current class.");}

      // the feature indices required by this weak machine
      virtual blitz::Array<int32_t,1> getIndices() const = 0;

      // machine IO
      virtual void save(bob::io::base::HDF5File& file) const = 0;
      virtual void load(bob::io::base::HDF5File& file) = 0;

    protected:
      WeakMachine(){}
      virtual ~WeakMachine(){}
  };

} } } // namespaces

#endif // BOB_LEARN_BOOSTING_WEAK_MACHINE_H
