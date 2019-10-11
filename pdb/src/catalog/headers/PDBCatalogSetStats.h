#pragma once

namespace pdb {

class PDBCatalogSetStats;
using PDBCatalogSetStatsPtr = std::shared_ptr<PDBCatalogSetStats>;

/**
 * This class is there to hold the aggregated stats about a set.
 */
class PDBCatalogSetStats {
public:

  PDBCatalogSetStats() = default;

  PDBCatalogSetStats(uint64_t numKeys, uint64_t numRecords, uint64_t setSize, uint64_t keySize)
      : numKeys(numKeys), numRecords(numRecords), setSize(setSize), keySize(keySize) {}

  /**
   * The number of keys in the set
   */
  uint64_t numKeys = 0;

  /**
   * The number of records in the set
   */
  uint64_t numRecords = 0;

  /**
   * The size of the set
   */
  uint64_t setSize = 0;

  /**
   * The size of the keys
   */
  uint64_t keySize = 0;

};


}
