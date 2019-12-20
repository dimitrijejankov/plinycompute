#include <utility>

#pragma once

#include <PDBCommunicator.h>
#include <PDBPageHandle.h>
#include <condition_variable>
#include <unordered_set>
#include <queue>
#include "../../../../applications/TestMatrixMultiply/sharedLibraries/headers/MatrixBlock.h"

namespace pdb {

class JoinAggSideSender {
public:

  using recordHandle = pdb::Handle<pdb::matrix::MatrixBlock>;


  JoinAggSideSender(PDBPageHandle page, PDBCommunicatorPtr  comm) : page(std::move(page)),
                                                                    communicator(std::move(comm)) {}

  /**
   * Takes in the records and sends them to a particular node
   * @param records - the records we want to send
   * @return the identifier which is passed to waitToFinish
   */
  int32_t queueToSend(std::vector<std::pair<uint32_t, recordHandle>> *records);

  /**
   * Wait to finish sending the records
   * @param id - the id of the records we are waiting
   */
  void waitToFinish(int32_t id);

  /**
   * run the sender
   */
  void run();

  /**
   * shutdown the sender
   */
  void shutdown();

private:

  /**
   * This indicates whether we are sending it
   */
  bool stillSending = true;

  /**
   * the queue of record vectors with their identifiers
   */
  std::queue<std::pair<int32_t, std::vector<std::pair<uint32_t, recordHandle>>*>> toSend;

  /**
   * the identifiers that are currently queued
   */
   std::unordered_set<int32_t> queuedIdentifiers;

   /**
    * the next id we are going to assign
    */
   int32_t nextID = 0;

  /**
   * The mutex to sync the sending
   */
  std::mutex m;

  /**
   * The conditional variable for waiting
   */
  std::condition_variable cv;

  /**
   * This is the page we are putting the stuff we want to send to
   */
   PDBPageHandle page;

  /**
   * This thing is sending our stuff to the right node
   */
  PDBCommunicatorPtr communicator;

};

// make a shared ptr shortcut
using JoinAggSideSenderPtr = std::shared_ptr<JoinAggSideSender>;

}