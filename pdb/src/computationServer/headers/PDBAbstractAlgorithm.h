//
// Created by dimitrije on 2/25/19.
//

#ifndef PDB_PDBPHYSICALALGORITHM_H
#define PDB_PDBPHYSICALALGORITHM_H


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

class PDBAbstractAlgorithm {
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
