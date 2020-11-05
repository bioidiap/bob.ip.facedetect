#include <bob.learn.boosting/StumpMachine.h>
#include <bob.core/cast.h>
#include <set>

bob::learn::boosting::StumpMachine::StumpMachine(double threshold, double polarity, int index):
  m_threshold(threshold),
  m_polarity(polarity),
  m_index(index)
{
}

bob::learn::boosting::StumpMachine::StumpMachine(bob::io::base::HDF5File& file):
  m_threshold(0),
  m_polarity(0),
  m_index(0)
{
  load(file);
}

double bob::learn::boosting::StumpMachine::_predict(double f) const{
  return m_polarity * ((-2. * (f < m_threshold)) + 1.);
}

double bob::learn::boosting::StumpMachine::forward(const blitz::Array<double, 1>& features) const{
  return _predict(features((int)m_index));
}

void bob::learn::boosting::StumpMachine::forward(const blitz::Array<double, 2>& features, blitz::Array<double,1> predictions) const{
  for (int i = features.extent(0); i--;){
    predictions(i) = _predict(features(i, (int)m_index));
  }
}

void bob::learn::boosting::StumpMachine::forward(const blitz::Array<double, 2>& features, blitz::Array<double,2> predictions) const{
  for (int i = features.extent(0); i--;){
    predictions(i,0) = _predict(features(i, (int)m_index));
  }
}


double bob::learn::boosting::StumpMachine::forward(const blitz::Array<uint16_t, 1>& features) const{
  return _predict(features((int)m_index));
}

void bob::learn::boosting::StumpMachine::forward(const blitz::Array<uint16_t, 2>& features, blitz::Array<double,1> predictions) const{
  for (int i = features.extent(0); i--;){
    predictions(i) = _predict(features(i, (int)m_index));
  }
}

void bob::learn::boosting::StumpMachine::forward(const blitz::Array<uint16_t, 2>& features, blitz::Array<double,2> predictions) const{
  for (int i = features.extent(0); i--;){
    predictions(i,0) = _predict(features(i, (int)m_index));
  }
}


blitz::Array<int32_t,1> bob::learn::boosting::StumpMachine::getIndices() const{
  blitz::Array<int32_t, 1> ret(1);
  ret = m_index;
  return ret;
}

void bob::learn::boosting::StumpMachine::load(bob::io::base::HDF5File& file){
  m_threshold = file.read<double>("Threshold");
  m_polarity = file.read<double>("Polarity");
  m_index = file.read<int32_t>("Index");
}

void bob::learn::boosting::StumpMachine::save(bob::io::base::HDF5File& file) const{
  file.set("Threshold", m_threshold);
  file.set("Polarity", m_polarity);
  file.set("Index", m_index);
  file.setAttribute(".", "MachineType", std::string("StumpMachine"));
}
