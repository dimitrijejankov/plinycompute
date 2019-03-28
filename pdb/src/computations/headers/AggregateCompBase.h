//
// Created by dimitrije on 3/27/19.
//

#ifndef PDB_AGGREGATECOMPBASE_H
#define PDB_AGGREGATECOMPBASE_H

#include "Computation.h"

namespace pdb {

class AggregateCompBase : public Computation {
public:

  virtual ComputeSinkPtr getAggregationHashMapCombiner(size_t workerID) = 0;

};

}

#endif //PDB_AGGREGATECOMPBASE_H
