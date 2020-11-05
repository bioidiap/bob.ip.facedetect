#include <bob.learn.boosting/JesorskyLoss.h>
#include <math.h>


static inline double sqr(const double x){
  return x*x;
}

double bob::learn::boosting::JesorskyLoss::interEyeDistance(const double y1, const double x1, const double y2, const double x2) const{
  return sqrt(sqr(y1 - y2) + sqr(x1 - x2));
}

void bob::learn::boosting::JesorskyLoss::loss(const blitz::Array<double, 2>& targets, const blitz::Array<double, 2>& scores, blitz::Array<double, 2>& errors) const{
  // compute one error for each sample
  errors = 0.;
  for (int i = targets.extent(0); i--;){
    // compute inter-eye-distance
    double scale = 1./interEyeDistance(targets(i,0), targets(i,1), targets(i,2), targets(i,3));
    // compute error for all positions
    // which are assumed to be 2D points
    for (int j = 0; j < targets.extent(1); j += 2){
      double dx = scores(i, j) - targets(i, j);
      double dy = scores(i, j+1) - targets(i, j+1);
      // sum errors
      errors(i,0) += sqrt(sqr(dx) + sqr(dy)) * scale;
    }
  }
}

void bob::learn::boosting::JesorskyLoss::lossGradient(const blitz::Array<double, 2>& targets, const blitz::Array<double, 2>& scores, blitz::Array<double, 2>& gradient) const{
//    # allocate memory for the gradients
//    gradient = numpy.ndarray(targets.shape, numpy.float)
  for (int i = targets.extent(0); i--;){
    // compute inter-eye-distance
    double scale = 1./interEyeDistance(targets(i,0), targets(i,1), targets(i,2), targets(i,3));
    // compute error for all positions
    // which are assumed to be 2D points
    for (int j = 0; j < targets.extent(1); j += 2){
      double dx = scores(i, j) - targets(i, j);
      double dy = scores(i, j+1) - targets(i, j+1);
      double error = scale / sqrt(sqr(dx) + sqr(dy));
      // set gradient
      gradient(i, j) = dx * error;
      gradient(i, j+1) = dy * error;
    }
  }
}


