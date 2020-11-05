#include <bob.learn.boosting/LossFunction.h>
#include <math.h>

void bob::learn::boosting::LossFunction::lossSum(const blitz::Array<double,1>& alpha, const blitz::Array<double,2>& targets, const blitz::Array<double,2>& previous_scores, const blitz::Array<double,2>& current_scores, blitz::Array<double,1>& loss_sum) const{
  // compute the scores and loss for the current alpha
  scores.resize(targets.shape());
  // TODO: is there any faster way for this?
  for (int i = scores.extent(0); i--;){
    for (int j = scores.extent(1); j--;){
      scores(i,j) = previous_scores(i,j) + alpha(j) * current_scores(i,j);
    }
  }
  errors.resize(targets.extent(0), 1);
  loss(targets, scores, errors);

  // compute the sum of the loss
  blitz::firstIndex i;
  blitz::secondIndex j;
  loss_sum = blitz::sum(errors(j,i), j);
}


void bob::learn::boosting::LossFunction::gradientSum(const blitz::Array<double,1>& alpha, const blitz::Array<double,2>& targets, const blitz::Array<double,2>& previous_scores, const blitz::Array<double,2>& current_scores, blitz::Array<double,1>& gradient_sum) const{
  // compute the scores and gradient for the current alpha
  scores.resize(targets.shape());
  // TODO: is there any faster way for this?
  for (int i = scores.extent(0); i--;){
    for (int j = scores.extent(1); j--;){
      scores(i,j) = previous_scores(i,j) + alpha(j) * current_scores(i,j);
    }
  }
//  scores = previous_scores + alpha * current_scores;

  gradients.resize(targets.shape());
  lossGradient(targets, scores, gradients);

  // take the sum of the loss gradient values
  const blitz::Array<double, 2> grad(gradients * current_scores);
  blitz::firstIndex i;
  blitz::secondIndex j;
  gradient_sum = blitz::sum(grad(j,i), j);
}


