//
// Created by dimitrije on 3/2/19.
//

#ifndef PDB_PDBSINKPAGESETSPEC_H
#define PDB_PDBSINKPAGESETSPEC_H

namespace pdb {

enum PDBSinkType {
  NoSink,
  SetSink,
  AggregationSink,
  AggShuffleSink,
  JoinShuffleSink,
  JoinShuffleIntermediateSink,
  BroadcastJoinSink,
  BroadcastIntermediateJoinSink,
};

// PRELOAD %PDBSinkPageSetSpec%

struct PDBSinkPageSetSpec : public Object  {
public:

  ENABLE_DEEP_COPY

  /**
   * The type of the sink
   */
  PDBSinkType sinkType;

  /**
   * Each page set is identified by a integer and a string. Generally set to (computationID, tupleSetIdentifier)
   * but relying on that is considered bad practice
   */
  std::pair<size_t, pdb::String> pageSetIdentifier;
};

}

#endif //PDB_PDBSINKPAGESETSPEC_H
