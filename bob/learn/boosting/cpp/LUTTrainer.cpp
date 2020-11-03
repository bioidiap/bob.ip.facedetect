#include <bob.learn.boosting/LUTTrainer.h>
#include <bob.learn.boosting/Functions.h>
#include <limits>

bob::learn::boosting::LUTTrainer::LUTTrainer(uint16_t maximumFeatureValue, int numberOfOutputs, SelectionStyle selectionType) :
  m_maximumFeatureValue(maximumFeatureValue),
  m_numberOfOutputs(numberOfOutputs),
  m_selectionType(selectionType),
  _luts(maximumFeatureValue, numberOfOutputs),
  _selectedIndices(numberOfOutputs),
  _gradientHistogram(maximumFeatureValue)
{
}

int32_t bob::learn::boosting::LUTTrainer::bestIndex(const blitz::Array<double,1>& array) const{
  double min = std::numeric_limits<double>::max();
  int32_t minIndex = -1;
  for (int i = 0; i < array.extent(0); ++i){
    if (array(i) < min){
      min = array(i);
      minIndex = i;
    }
  }
  return minIndex;
}

void bob::learn::boosting::LUTTrainer::weightedHistogram(const blitz::Array<uint16_t,1>& features, const blitz::Array<double,1>& weights) const{
  bob::core::array::assertSameShape(features, weights);
  _gradientHistogram = 0.;
  for (int i = features.extent(0); i--;){
    _gradientHistogram((int)features(i)) += weights(i);
  }
}

boost::shared_ptr<bob::learn::boosting::LUTMachine> bob::learn::boosting::LUTTrainer::train(const blitz::Array<uint16_t,2>& trainingFeatures, const blitz::Array<double,2>& lossGradient) const{
  int featureLength = trainingFeatures.extent(1);
  _lossSum.resize(featureLength, m_numberOfOutputs);
  // Compute the sum of the gradient based on the feature values or the loss associated with each feature index
  // Compute the loss for each feature
  for (int featureIndex = featureLength; featureIndex--;){
    for (int outputIndex = m_numberOfOutputs; outputIndex--;){
      weightedHistogram(trainingFeatures(blitz::Range::all(),featureIndex), lossGradient(blitz::Range::all(), outputIndex));
      _lossSum(featureIndex,outputIndex) = - blitz::sum(blitz::abs(_gradientHistogram));
    }
  }

  // Select the most discriminative index (or indices) for classification which minimizes the loss
  //  and compute the sum of gradient for that index
  if (m_selectionType == independent){
    // independent feature selection is used if all the dimension of output use different feature
    // each of the selected feature minimize a dimension of the loss function
    for (int outputIndex = m_numberOfOutputs; outputIndex--;){
      _selectedIndices(outputIndex) = bestIndex(_lossSum(blitz::Range::all(),outputIndex));
    }
  } else {
    // for 'shared' feature selection the loss function is summed over multiple dimensions and
    // the feature that minimized this cumulative loss is used for all the outputs
    blitz::secondIndex j;
    const blitz::Array<double,1> sum(blitz::sum(_lossSum, j));
    _selectedIndices = bestIndex(sum);
  }

  // compute the look-up-tables for the best index
  for (int outputIndex = m_numberOfOutputs; outputIndex--;){
    int selectedIndex = _selectedIndices(outputIndex);
    weightedHistogram(trainingFeatures(blitz::Range::all(), selectedIndex), lossGradient(blitz::Range::all(), outputIndex));

    for (int lutIndex = m_maximumFeatureValue; lutIndex--;){
      _luts(lutIndex, outputIndex) = (_gradientHistogram(lutIndex) > 0) * 2. - 1.;
    }
  }

  // create new weak machine
  return boost::shared_ptr<LUTMachine>(new LUTMachine(_luts.copy(), _selectedIndices.copy()));

}
