#pragma once


// PRELOAD %CatSetStatsResult%

namespace pdb {

// encapsulates a request to obtain a the stats of a set from a catalog
class CatSetStatsResult : public Object {

public:

  CatSetStatsResult(bool has_stats, uint64_t record_count, uint64_t shard_size, uint64_t key_count, uint64_t key_size)
      : hasStats(has_stats), recordCount(record_count), shardSize(shard_size), keyCount(key_count), keySize(key_size) {}

  CatSetStatsResult() = default;
  ~CatSetStatsResult() = default;

  ENABLE_DEEP_COPY

  /**
   * Does it have any stats in the first place
   */
  bool hasStats = false;

  /**
   * The record count
   */
  uint64_t recordCount = 0;

  /**
   * The size of the shard on this node in bytes
   */
  uint64_t shardSize = 0;

  /**
   * The key count on this node, a node always holds all the keys
   */
  uint64_t keyCount = 0;

  /**
   * The size of the keys on this node, a node always holds all the keys
   */
  uint64_t keySize = 0;
};

}
