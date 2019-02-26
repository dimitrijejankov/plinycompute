//
// Created by dimitrije on 2/25/19.
//

#ifndef PDB_PDBPHYSICALALGORITHM_H
#define PDB_PDBPHYSICALALGORITHM_H

#include "SourcePageSetSpec.h"
#include "SinkPageSetSpec.h"

namespace pdb {

enum PDBAbstractAlgorithmType {

  ShuffleForJoin,
  BroadcastForJoin,
  DistributedAggregation,
  StraightPipe
};

enum PDBSourceType {
  MergeSource,
  SetScanSource,
  AggregationSource,
  ShuffledAggregatesSource,
  ShuffledJoinTuplesSource,
  BroadcastJoinSource
};

enum PDBSinkType {
  SetSink,
  AggregationSink,
  AggShuffleSink,
  JoinShuffleSink,
  BroadcastJoinSink
};

class PDBAbstractAlgorithm { // Shouldn't this inherit from pdb::Object?
public:

  /**
   * Sets up the whole algorithm
   */
  virtual void setup() = 0;

  /**
   * Runs the algorithm
   */
  virtual void run() = 0;

  /**
   * Returns the type of the algorithm we want to run
   */
  virtual PDBAbstractAlgorithmType getAlgorithmType() = 0;

private:

  /**
   * The input page set specifiers. Note that a PDBAbstractAlgorithm, even a StraightPipe,
   * can have multiple page sets as input.
   */
  Handle<Vector<Handle<SourcePageSetSpec>>> sourceTypes;

  /**
   * The output page set specifier. There can only be a single PageSet as output.
   */
  Handle<SinkPageSetSpec> sinkType;

};

}

#endif //PDB_PDBPHYSICALALGORITHM_H
