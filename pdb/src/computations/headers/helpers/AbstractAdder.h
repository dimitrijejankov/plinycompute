//
// Created by vicram on 6/15/19.
//

#ifndef PDB_ABSTRACTADDER_H
#define PDB_ABSTRACTADDER_H

namespace pdb {

/**
 * This is the interface for a class which has a single method, 'add'.
 * @tparam In1 Type of the first input.
 * @tparam In2 Type of the second input; defaults to In1.
 * @tparam Out Type of the object that results from adding In1 and In2. Defaults to In1.
 */
template<class In1, class In2=In1, class Out=In1>
class AbstractAdder {
 public:
  virtual Out add(In1& in1, In2& in2) = 0;
};

}

#endif //PDB_ABSTRACTADDER_H
