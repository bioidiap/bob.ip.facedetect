#ifndef BOB_LEARN_BOOSTING_LUT_MACHINE_H
#define BOB_LEARN_BOOSTING_LUT_MACHINE_H

#include <bob.learn.boosting/WeakMachine.h>

namespace bob { namespace learn { namespace boosting {

  /**
   * This machine uses a Look-Up-Table-based decision using *discrete* features.
   *
   * For each discrete value of the feature, either +1 or -1 is returned.
   * This machine can be used in a multi-variate environment.
   */
  class LUTMachine : public WeakMachine{
    public:
      // Create an LUT machine using the given LUT and the given index
      LUTMachine(const blitz::Array<double,1> look_up_table, int index);
      // Create an LUT machine using the given LUTs for each output dimension, and the corresponding indices into the feature vector
      LUTMachine(const blitz::Array<double,2> look_up_tables, const blitz::Array<int,1> indices);
      // Creates an LUT machine from file
      LUTMachine(bob::io::base::HDF5File& file);

      // uni-variate single-feature classification of the input feature vector
      virtual double forward(const blitz::Array<uint16_t, 1>& features) const;
      // multi-variate single-feature classification of the input feature vector
      virtual void forward(const blitz::Array<uint16_t, 1>& features, blitz::Array<double,1> predictions) const;
      // uni-variate classification of several input feature vector
      virtual void forward(const blitz::Array<uint16_t, 2>& features, blitz::Array<double,1> predictions) const;
      // multi-variate classification of several input feature vector
      virtual void forward(const blitz::Array<uint16_t, 2>& features, blitz::Array<double,2> predictions) const;

      // The indices into the feature vector used by this machine
      virtual blitz::Array<int32_t,1> getIndices() const;

      // machine IO
      virtual void save(bob::io::base::HDF5File& file) const;
      virtual void load(bob::io::base::HDF5File& file);

      // The multi-variate look-up-table used in this machine
      const blitz::Array<double, 2> getLut() const{return m_look_up_tables;}

    private:
      // the LUT for the multi-variate case
      blitz::Array<double,2> m_look_up_tables;
      // The feature indices used in each of the output dimensions
      blitz::Array<int32_t,1> m_indices;

      // for speed reasons, we also keep the LUT for the uni-variate case
      blitz::Array<double,1> _look_up_table;
      // and the index
      int32_t _index;
  };

} } } // namespaces

#endif // BOB_LEARN_BOOSTING_LUT_MACHINE_H
