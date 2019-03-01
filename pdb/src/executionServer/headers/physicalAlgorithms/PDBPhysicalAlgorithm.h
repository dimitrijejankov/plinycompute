//
// Created by dimitrije on 2/25/19.
//

#ifndef PDB_PDBPHYSICALALGORITHM_H
#define PDB_PDBPHYSICALALGORITHM_H

#include <Object.h>
#include <PDBString.h>

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

struct SourcePageSetSpec {

  /**
   * The computation that is consuming or producing this page set
   */
  pdb::String tupleSetIdentifier;

  /**
   *
   */
  PDBSourceType sourceType;

  /**
   * Each page set is identified by a integer and a string. Generally set to (computationID, tupleSetIdentifier)
   * but relying on that is considered bad practice
   */
  std::pair<size_t, pdb::String> pageSetIdentifier;
};

struct SinkPageSetSpec {

  /**
   * The computation that is consuming or producing this page set
   */
  pdb::String tupleSetIdentifier;

  /**
   *
   */
  PDBSinkType sinkType;

  /**
   * Each page set is identified by a integer and a string. Generally set to (computationID, tupleSetIdentifier)
   * but relying on that is considered bad practice
   */
  std::pair<size_t, pdb::String> pageSetIdentifier;
};


class PDBPhysicalAlgorithm : public Object {
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
   * The source type the algorithm should setup
   */
  PDBSourceType sourceType;

  /**
   * The sink type the algorithm should setup
   */
  PDBSinkType sinkType;

};

}

#endif //PDB_PDBPHYSICALALGORITHM_H
