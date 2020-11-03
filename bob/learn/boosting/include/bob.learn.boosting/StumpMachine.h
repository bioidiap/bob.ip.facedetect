#ifndef BOB_LEARN_BOOSTING_STUMP_MACHINE_H
#define BOB_LEARN_BOOSTING_STUMP_MACHINE_H


#include <bob.learn.boosting/WeakMachine.h>

namespace bob { namespace learn { namespace boosting {

  /**
   * Implements a decision stump based weak machine, e.g., as used by the Viola-Jones face detector.
   *
   * This machine only allows uni-variate decision (yes/no) for the given input feature(s).
   */
  class StumpMachine : public WeakMachine{
    public:
      // Create a decision stump machine with the given threshold, the given polarity (+-1) and the index into the feature vector
      StumpMachine(double threshold, double polarity, int index);
      // Create a decision stump machine from file
      StumpMachine(bob::io::base::HDF5File& file);
      virtual ~StumpMachine(){}

      // forwarding of a single feature
      virtual double forward(const blitz::Array<uint16_t, 1>& features) const;
      virtual double forward(const blitz::Array<double, 1>& features) const;

      // forwarding of multiple features
      virtual void forward(const blitz::Array<double, 2>& features, blitz::Array<double,1> predictions) const;
      virtual void forward(const blitz::Array<uint16_t, 2>& features, blitz::Array<double,1> predictions) const;

      // forwarding of multiple features
      virtual void forward(const blitz::Array<double, 2>& features, blitz::Array<double,2> predictions) const;
      virtual void forward(const blitz::Array<uint16_t, 2>& features, blitz::Array<double,2> predictions) const;

      // the index used by this machine
      virtual blitz::Array<int32_t,1> getIndices() const;

      // the threshold
      double getThreshold() const {return m_threshold;}
      // the polarity (i.e., does a lower or higher value correspond to the positive class?)
      double getPolarity() const {return m_polarity;}

      // Machine IO
      virtual void save(bob::io::base::HDF5File& file) const;
      virtual void load(bob::io::base::HDF5File& file);

    private:
      // helper function to compute the prediction for a single feature value
      double _predict(double f) const;

      // the data used by this class:
      double m_threshold;
      double m_polarity;
      int32_t m_index;
  };

} } } // namespaces

#endif // BOB_LEARN_BOOSTING_STUMP_MACHINE_H
