#ifndef BOB_LEARN_BOOSTING_BOOSTED_MACHINE_H
#define BOB_LEARN_BOOSTING_BOOSTED_MACHINE_H

#include <bob.io.base/HDF5File.h>

#include <bob.learn.boosting/WeakMachine.h>

namespace bob { namespace learn { namespace boosting {

  class BoostedMachine{
    public:
      BoostedMachine();
      BoostedMachine(bob::io::base::HDF5File& file);

      // adds the uni-variate weak machine with the given weight
      void add_weak_machine(const boost::shared_ptr<WeakMachine> weak_machine, const double weight);
      // adds the multi-variate weak machine with the given weights per output
      void add_weak_machine(const boost::shared_ptr<WeakMachine> weak_machine, const blitz::Array<double,1> weights);

      // predicts the output for the given single feature
      double forward(const blitz::Array<uint16_t, 1>& features) const;

      // predicts the output for the given single feature (multi-variate case)
      void forward(const blitz::Array<uint16_t, 1>& features, blitz::Array<double,1> predictions) const;

      // predicts the output for multiple features (uni-variate case)
      void forward(const blitz::Array<uint16_t, 2>& features, blitz::Array<double,1> predictions) const;

      // predicts the output for multiple features (multi-variate case)
      void forward(const blitz::Array<uint16_t, 2>& features, blitz::Array<double,2> predictions) const;

      // predicts the output and the labels for the given features (uni-variate case)
      void forward(const blitz::Array<uint16_t, 2>& features, blitz::Array<double,1> predictions, blitz::Array<double,1> labels) const;

      // predicts the output and the labels for the given features (multi-variate case)
      void forward(const blitz::Array<uint16_t, 2>& features, blitz::Array<double,2> predictions, blitz::Array<double,2> labels) const;

      // the number of outputs of the machine (multi-variate); 1 for the uni-variate case
      int numberOfOutputs() const {return m_weights.extent(1);}

      // computed the sorted unique list of indices used by the weak machines in the given range of machines
      blitz::Array<int32_t,1> getIndices(int start = 0, int end = -1) const;

      // returns the weights of the machines (multi-variate)
      const blitz::Array<double,2> getWeights() const {return m_weights;}

      // returns the weak machines
      const std::vector<boost::shared_ptr<WeakMachine> >& getWeakMachines() const {return m_weak_machines;}

      // writes the machine to file
      void save(bob::io::base::HDF5File& file) const;

      // loads the machine from file
      void load(bob::io::base::HDF5File& file);


    private:
      // The weak machines
      std::vector<boost::shared_ptr<WeakMachine> > m_weak_machines;
      // the (multi-variate) weights of the machines
      blitz::Array<double,2> m_weights;
      // a shortcut to speed up uni-variate access
      blitz::Array<double,1> _weights;

      // shortcut to avoid allocating memory for each call of 'forward'
      mutable blitz::Array<double,1> _predictions1;
      mutable blitz::Array<double,2> _predictions2;
  };

} } } // namespaces

#endif // BOB_LEARN_BOOSTING_BOOSTED_MACHINE_H
