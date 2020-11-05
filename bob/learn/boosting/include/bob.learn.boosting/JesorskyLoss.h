#ifndef BOB_LEARN_BOOSTING_JESORSKY_LOSS_H
#define BOB_LEARN_BOOSTING_JESORSKY_LOSS_H

#include <blitz/array.h>
#include <bob.learn.boosting/LossFunction.h>

namespace bob { namespace learn { namespace boosting {

  /**
   * This machine uses a Look-Up-Table-based decision using *discrete* features.
   *
   * For each discrete value of the feature, either +1 or -1 is returned.
   * This machine can be used in a multi-variate environment.
   */
  class JesorskyLoss : public LossFunction{
    public:
      // Create an LUT machine using the given LUT and the given index
      JesorskyLoss(){}
      virtual ~JesorskyLoss(){}

      void loss(const blitz::Array<double, 2>& targets, const blitz::Array<double, 2>& scores, blitz::Array<double, 2>& errors) const;

      void lossGradient(const blitz::Array<double, 2>& targets, const blitz::Array<double, 2>& scores, blitz::Array<double, 2>& gradient) const;

    private:

      double interEyeDistance(const double y1, const double x1, const double y2, const double x2) const;
  };

} } } // namespaces

#endif // BOB_LEARN_BOOSTING_JESORSKY_LOSS_H
