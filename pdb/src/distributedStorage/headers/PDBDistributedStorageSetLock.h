#pragma once

#include <memory>

namespace pdb {

// predefine storage to avoid circular dependency
class PDBDistributedStorage;
using PDBDistributedStoragePtr = std::shared_ptr<PDBDistributedStorage>;

/**
 * This enum indicates what state the set currently is
 */
enum class PDBDistributedStorageSetState {

  NONE,
  WRITING_DATA,
  READING_DATA,
  WRITE_READ_DATA,
  CLEARING_DATA,
  Default = NONE
};

class PDBDistributedStorageSetLock;
using PDBDistributedStorageSetLockPtr = std::shared_ptr<PDBDistributedStorageSetLock>;

class PDBDistributedStorageSetLock {

public:

  PDBDistributedStorageSetLock(const std::string &dbName,
                               const std::string &setName,
                               pdb::PDBDistributedStorageSetState state,
                               const pdb::PDBDistributedStoragePtr &storage);

  /**
   * Releases the request
   */
  ~PDBDistributedStorageSetLock();

  /**
   * Returns true if a write request was granted
   * @return true if it was false otherwise
   */
  bool isWriteGranted();

  /**
   * Returns true if a read request was granted
   * @return true if it was false otherwise
   */
  bool isReadGranted();

  /**
   * Returns true if a clear request was granted
   * @return true if it was false otherwise
   */
  bool isClearGranted();

  /**
   * Returns if any access is grated
   * @return true if it was false otherwise
   */
  bool isGranted();

private:

  /**
   * The state of the storage
   */
  PDBDistributedStorageSetState state;

  /**
   * A pointer to the distributed storage that granted the lock
   */
  PDBDistributedStoragePtr storage;

  /**
   * The name of the database the set belongs to
   */
  std::string dbName;

  /**
   * The name of the set
   */
  std::string setName;
};

}
