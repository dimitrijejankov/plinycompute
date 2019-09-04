#pragma once

namespace pdb {

enum PDBSourceType {
  NoSource,
  MergeSource,
  SetScanSource,
  AggregationSource,
  ShuffledAggregatesSource,
  ShuffledJoinTuplesSource,
  JoinedShuffleSource,
  BroadcastJoinSource,
  BroadcastIntermediateJoinSource
};

// PRELOAD %PDBSourcePageSetSpec%

struct PDBSourcePageSetSpec : public Object {

public:

  ENABLE_DEEP_COPY

  /**
   * The type of the source
   */
  PDBSourceType sourceType;

  /**
   * Each page set is identified by a integer and a string. Generally set to (computationID, tupleSetIdentifier)
   * but relying on that is considered bad practice
   */
  std::pair<size_t, pdb::String> pageSetIdentifier;
};

}