#ifndef BOB_LEARN_BOOSTING_LUT_TRAINER_H
#define BOB_LEARN_BOOSTING_LUT_TRAINER_H

#include <bob.learn.boosting/LUTMachine.h>


namespace bob { namespace learn { namespace boosting {

  /**
   * This machine uses a Look-Up-Table-based decision using *discrete* features.
   *
   * For each discrete value of the feature, either +1 or -1 is returned.
   * This machine can be used in a multi-variate environment.
   */
  class LUTTrainer{
    public:
      typedef enum {
        independent = 0,
        shared = 1
      } SelectionStyle;

      // Create an LUT machine using the given LUT and the given index
      LUTTrainer(uint16_t maximumFeatureValue, int numberOfOutputs = 1, SelectionStyle selectionType = independent);

      boost::shared_ptr<LUTMachine> train(const blitz::Array<uint16_t, 2>& training_features, const blitz::Array<double,2>& loss_gradient) const;

      uint16_t maximumFeatureValue() const {return m_maximumFeatureValue;}
      int numberOfOutputs() const {return m_numberOfOutputs;}
      SelectionStyle selectionType() const {return m_selectionType;}

    private:
      int32_t bestIndex(const blitz::Array<double,1>& array) const;
      void weightedHistogram(const blitz::Array<uint16_t,1>& features, const blitz::Array<double,1>& weights) const;

      uint16_t m_maximumFeatureValue;
      int m_numberOfOutputs;
      SelectionStyle m_selectionType;

      // pre-allocated arrays for faster access
      mutable blitz::Array<double,2> _luts;
      mutable blitz::Array<int32_t,1> _selectedIndices;
      mutable blitz::Array<double,1> _gradientHistogram;
      mutable blitz::Array<double,2> _lossSum;

  };

} } } // namespaces

#endif // BOB_LEARN_BOOSTING_LUT_TRAINER_H
