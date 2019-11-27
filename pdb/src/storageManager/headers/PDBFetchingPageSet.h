#pragma once

#include <thread>
#include <PDBWorker.h>
#include <PageProcessor.h>
#include "PDBAbstractPageSet.h"
#include "PDBFetchingPageSet.h"
#include "PDBCommunicator.h"
#include "PDBStorageManagerBackend.h"
#include <PDBBufferManagerInterface.h>

namespace pdb {

class PDBFetchingPageSet : public PDBAbstractPageSet {
public:

  explicit PDBFetchingPageSet(PDBCommunicatorPtr communicator,
                              PDBStorageManagerBackendPtr storageManager,
                              PDBBufferManagerInterfacePtr bufferManager,
                              uint64_t numPages);

  ~PDBFetchingPageSet();

  /**
   *
   * @param workerID
   * @return
   */
  PDBPageHandle getNextPage(size_t workerID) override;

  /**
   *
   * @return
   */
  PDBPageHandle getNewPage() override;

  /**
   *
   * @return
   */
  size_t getNumPages() override;

  void removePage(PDBPageHandle pageHandle) override;
  size_t getMaxPageSize() override;

  /**
   *
   */
  void resetPageSet() override;

protected:

  /**
   * The communicator we use to fetch the pages
   */
  PDBCommunicatorPtr communicator;

  /**
   * The buffer manger so we can grab empty pages
   */
  PDBBufferManagerInterfacePtr bufferManager;

  /**
   * The storage manager
   */
  PDBStorageManagerBackendPtr storageManager;

  /**
   * This thread is grabbing the pages from the node
   */
  PDBWorkerPtr worker;

  /**
   * The buzzer of the fetcher thread
   */
  PDBBuzzerPtr tempBuzzer;

  /**
   * The page queue where the worker is going to be putting pages...
   */
  PDBPageQueuePtr queue;

  /**
   * How many pages do we have
   */
  uint64_t numPages = 0;

  /**
   * The current page
   */
  uint64_t curPage = 0;

  /**
   * The success
   */
  bool success = true;

  /**
   * The error the page set encountered
   */
  std::string error;
};

}