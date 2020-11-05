#include <bob.learn.boosting/BoostedMachine.h>
#include <bob.learn.boosting/Functions.h>
#include <sstream>
#include <set>

bob::learn::boosting::BoostedMachine::BoostedMachine() :
  m_weak_machines(),
  m_weights()
{
}

bob::learn::boosting::BoostedMachine::BoostedMachine(bob::io::base::HDF5File& file) :
  m_weak_machines(),
  m_weights()
{
  load(file);
}

void bob::learn::boosting::BoostedMachine::add_weak_machine(const boost::shared_ptr<WeakMachine> weak_machine, const double weight){
  m_weak_machines.push_back(weak_machine);
  m_weights.resizeAndPreserve(m_weak_machines.size(), 1);
  m_weights(m_weights.extent(0)-1, 0) = weight;
  _weights.reference(m_weights(blitz::Range::all(), 0));
}


void bob::learn::boosting::BoostedMachine::add_weak_machine(const boost::shared_ptr<WeakMachine> weak_machine, const blitz::Array<double,1> weights){
  m_weak_machines.push_back(weak_machine);
  m_weights.resizeAndPreserve(m_weak_machines.size(), weights.extent(0));
  m_weights(m_weights.extent(0)-1, blitz::Range::all()) = weights;
  _weights.reference(m_weights(blitz::Range::all(), 0));
}


double bob::learn::boosting::BoostedMachine::forward(const blitz::Array<uint16_t,1>& features) const{
  // univariate, single feature
  double sum = 0.;
  //TODO: optimize using STL
  for (int i = m_weak_machines.size(); i--;){
    sum += _weights(i) * m_weak_machines[i]->forward(features);
  }
  return sum;
}

void bob::learn::boosting::BoostedMachine::forward(const blitz::Array<uint16_t,1>& features, blitz::Array<double,1> predictions) const{
  // multi-variate, single feature
  // initialize the predictions since they will be overwritten
  _predictions1.resize(predictions.shape());
  predictions = 0.;
  for (int i = m_weak_machines.size(); i--;){
    // predict locally
    m_weak_machines[i]->forward(features, _predictions1);
    predictions(blitz::Range::all()) += m_weights(i, blitz::Range::all()) * _predictions1(blitz::Range::all());
  }
}

void bob::learn::boosting::BoostedMachine::forward(const blitz::Array<uint16_t,2>& features, blitz::Array<double,1> predictions) const{
  // univariate, multiple features
  // initialize the predictions since they will be overwritten
  _predictions1.resize(predictions.shape());
  predictions = 0.;
  for (int i = m_weak_machines.size(); i--;){
    // predict locally
    m_weak_machines[i]->forward(features, _predictions1);
    predictions(blitz::Range::all()) += _weights(i) * _predictions1(blitz::Range::all());
  }
}

void bob::learn::boosting::BoostedMachine::forward(const blitz::Array<uint16_t,2>& features, blitz::Array<double,2> predictions) const{
  // initialize the predictions since they will be overwritten
  _predictions2.resize(predictions.shape());
  predictions = 0.;
  for (int i = m_weak_machines.size(); i--;){
    // predict locally
    m_weak_machines[i]->forward(features, _predictions2);
    for (int j = predictions.extent(0); j--;)
      predictions(j, blitz::Range::all()) += m_weights(i, blitz::Range::all()) * _predictions2(j, blitz::Range::all());
  }
}


void bob::learn::boosting::BoostedMachine::forward(const blitz::Array<uint16_t,2>& features, blitz::Array<double,1> predictions, blitz::Array<double,1> labels) const{
  forward(features, predictions);
  // get the labels
  for (int i = predictions.extent(0); i--;)
    labels(i) = (predictions(i) > 0) * 2. - 1;
}

void bob::learn::boosting::BoostedMachine::forward(const blitz::Array<uint16_t,2>& features, blitz::Array<double,2> predictions, blitz::Array<double,2> labels) const{
  forward(features, predictions);
  // get the labels
  labels = -1;
  for (int i = predictions.extent(0); i--;){
    labels(i, blitz::maxIndex(predictions(i, blitz::Range::all()))[0]) = 1;
  }
}


blitz::Array<int,1> bob::learn::boosting::BoostedMachine::getIndices(int start, int end) const{
  std::set<int32_t> indices;
  if (end < 0) end = m_weak_machines.size();
  for (int i = start; i < end; ++i){
    const blitz::Array<int32_t,1>& ind = m_weak_machines[i]->getIndices();
    indices.insert(ind.begin(), ind.end());
  }

  blitz::Array<int32_t,1> ret(indices.size());
  std::copy(indices.begin(), indices.end(), ret.begin());
  return ret;
}

// writes the machine to file
void bob::learn::boosting::BoostedMachine::save(bob::io::base::HDF5File& file) const{
  file.setAttribute(".", "version", 2);
  file.setArray("Weights", m_weights);
  for (int i = 0; i < m_weights.extent(0); ++i){
    std::ostringstream fns;
    fns << "WeakMachine_" << i;
    file.createGroup(fns.str());
    file.cd(fns.str());
    m_weak_machines[i]->save(file);
    file.cd("..");
  }
}

// loads the machine from file
void bob::learn::boosting::BoostedMachine::load(bob::io::base::HDF5File& file){
  m_weak_machines.clear();

  // the weights
  m_weights.reference(file.readArray<double,2>("Weights"));
  _weights.reference(m_weights(blitz::Range::all(), 0));

  // name of the first machine
  std::string machine_name("WeakMachine_0");
  while (file.hasGroup(machine_name)){
    // load weight and machine
    file.cd(machine_name);
    m_weak_machines.push_back(loadWeakMachine(file));
    file.cd("..");
    // get name of the next machine
    std::ostringstream fns;
    fns << "WeakMachine_" << m_weak_machines.size();
    machine_name = fns.str();
  }

  if (m_weak_machines.empty()){
    throw std::runtime_error("Could not read weak machines.");
  }
}


