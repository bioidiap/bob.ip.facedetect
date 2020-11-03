#include <bob.learn.boosting/LUTMachine.h>

#include <bob.core/cast.h>
#include <bob.core/assert.h>
#include <set>

bob::learn::boosting::LUTMachine::LUTMachine(const blitz::Array<double,1> look_up_table, const int index):
  m_look_up_tables(look_up_table.extent(0), 1),
  m_indices(1,1),
  _look_up_table(),
  _index(index)
{
  // we have to copy the array, otherwise weird things happen
  m_look_up_tables(0, blitz::Range::all()) = look_up_table;
  m_indices(0) = index;
  // for the shortcut, we just reference the first row of the the look up tables
  _look_up_table.reference(m_look_up_tables(blitz::Range::all(),0));
  _index = m_indices(0);
}

bob::learn::boosting::LUTMachine::LUTMachine(const blitz::Array<double,2> look_up_tables, const blitz::Array<int,1> indices):
  m_look_up_tables(look_up_tables.shape()),
  m_indices(indices.shape()),
  _look_up_table(),
  _index(0)
{
  // we have to copy the array, otherwise weird things happen
  m_look_up_tables = look_up_tables;
  m_indices = indices;
  // for the shortcut, we just reference the first row of the the look up tables
  _look_up_table.reference(m_look_up_tables(blitz::Range::all(),0));
  _index = m_indices(0);
}

bob::learn::boosting::LUTMachine::LUTMachine(bob::io::base::HDF5File& file):
  m_look_up_tables(),
  m_indices(),
  _look_up_table(),
  _index(0)
{
  load(file);
}

double bob::learn::boosting::LUTMachine::forward(const blitz::Array<uint16_t,1>& features) const{
  // univariate, single feature
#ifdef BOB_DEBUG
  if ( features.extent(0) <= _index ) throw std::runtime_error((boost::format("The index %d of this machine is out of range %d")%_index%features.extent(0)).str());
  if ( features((int)_index) >= _look_up_table.extent(0) ) throw std::runtime_error((boost::format("The feature %d at index %d is out of range %d")%features((int)_index)%_index%_look_up_table.extent(0)).str());
#endif // BOB_DEBUG
  return _look_up_table((int)features(_index));
}


void bob::learn::boosting::LUTMachine::forward(const blitz::Array<uint16_t,1>& features, blitz::Array<double,1> predictions) const{
  // multi-variate, single feature
#ifdef BOB_DEBUG
  bob::core::array::assertSameShape(m_indices, predictions);
#endif // BOB_DEBUG

  for (int j = 0; j < m_indices.extent(0); ++j){
#ifdef BOB_DEBUG
    if ( features.extent(0) <= m_indices(j) ) throw std::runtime_error((boost::format("One of the indices %d of this machine is out of range %d")%m_indices(j)%features.extent(0)).str());
#endif // BOB_DEBUG

    predictions(j) = m_look_up_tables((int)features(m_indices(j)), j);
  }
}

void bob::learn::boosting::LUTMachine::forward(const blitz::Array<uint16_t,2>& features, blitz::Array<double,1> predictions) const{
  // univariate, several features
#ifdef BOB_DEBUG
  if ( predictions.extent(0) != features.extent(0) ) throw std::runtime_error((boost::format("The number of predictions must match the number of features, but they don't: %d != %d")%predictions.extent(0)%features.extent(0)).str());
  if ( features.extent(1) <= _index ) throw std::runtime_error((boost::format("The index %d of this machine is out of range %d")%_index%features.extent(1)).str());
#endif // BOB_DEBUG

  for (int i = features.extent(0); i--;){
#ifdef BOB_DEBUG
    if ( features(i, (int)_index) >= _look_up_table.extent(0) ) throw std::runtime_error((boost::format("The feature %d at index %d is out of range %d")%features(i, (int)_index)%_index%_look_up_table.extent(0)).str());
#endif // BOB_DEBUG

    predictions(i) = _look_up_table((int)features(i, _index));
  }
}

void bob::learn::boosting::LUTMachine::forward(const blitz::Array<uint16_t,2>& features, blitz::Array<double,2> predictions) const{
  // multi-variate, several features
#ifdef BOB_DEBUG
  if ( predictions.extent(0) != features.extent(0) ) throw std::runtime_error((boost::format("The number of predictions must match the number of features, but they don't: %d != %d")%predictions.extent(0)%features.extent(0)).str());
  if ( predictions.extent(1) != m_indices.extent(0) ) throw std::runtime_error((boost::format("The size of predictions must match the number of indices, but they don't: %d != %d")%predictions.extent(1)%m_indices.extent(0)).str());

  for (int j = m_indices.extent(0); j--;)
    if ( features.extent(1) <= m_indices(j) ) throw std::runtime_error((boost::format("One of the indices %d of this machine is out of range %d")%m_indices(j)%features.extent(1)).str());
#endif // BOB_DEBUG

  for (int i = 0; i < features.extent(0); ++i){
    for (int j = 0; j < m_indices.extent(0); ++j){
      predictions(i,j) = m_look_up_tables((int)features(i, m_indices(j)), j);
    }
  }
}

blitz::Array<int32_t,1> bob::learn::boosting::LUTMachine::getIndices() const{
  std::set<int32_t> indices;
  for (int i = 0; i < m_indices.extent(0); ++i){
    indices.insert(m_indices(i));
  }
  blitz::Array<int32_t, 1> ret(indices.size());
  std::copy(indices.begin(), indices.end(), ret.begin());
  return ret;
}

void bob::learn::boosting::LUTMachine::load(bob::io::base::HDF5File& file){
  try{
    m_look_up_tables.reference(file.readArray<double,2>("LUT"));
  }catch (std::exception){
    m_look_up_tables.reference(bob::core::array::cast<double>(file.readArray<int32_t,2>("LUT")));
  }
  try{
    m_indices.reference(file.readArray<int,1>("Indices"));
  }catch (std::exception){
    m_indices.reference(bob::core::array::cast<int>(file.readArray<int32_t,1>("Indices")));
  }

  _look_up_table.reference(m_look_up_tables(blitz::Range::all(), 0));
  _index = m_indices(0);
}

void bob::learn::boosting::LUTMachine::save(bob::io::base::HDF5File& file) const{
  file.setArray("LUT", m_look_up_tables);
  file.setArray("Indices", m_indices);
  file.setAttribute(".", "MachineType", std::string("LUTMachine"));
}
