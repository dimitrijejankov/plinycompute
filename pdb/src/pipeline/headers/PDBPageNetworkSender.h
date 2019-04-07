//
// Created by dimitrije on 4/5/19.
//

#ifndef PDB_PAGESENDER_H
#define PDB_PAGESENDER_H

#include <memory>
#include <PageProcessor.h>
#include <PDBCommunicator.h>

namespace pdb {

class PDBPageNetworkSender;
using PDBPageNetworkSenderPtr = std::shared_ptr<PDBPageNetworkSender>;

/**
 * This class sends pages over the wire. It gets pages from the provided queue and sends them over the wire.
 */
class PDBPageNetworkSender {
public:

  PDBPageNetworkSender(string address, int32_t port, uint64_t numberOfProcessingThreads, uint64_t numberOfNodes,
                       uint64_t maxRetries, PDBLoggerPtr logger, std::pair<uint64_t, std::string> pageSetID, pdb::PDBPageQueuePtr queue);

  /**
   * Connects to the node with the parameters provided in the constructor and gets the ACK that the other side has set everything up.
   * @return - true if we succeed false otherwise
   */
  bool setup();

  /**
   * Starts grabbing pages from the queue and sending them over the wire until we get a null ptr from the queue
   * @return true if everything works just fine false otherwise
   */
  bool run();

private:

  /**
   * The error if any
   */
  std::string errMsg;

  /**
   * The address of the node where we are sending the pages to
   */
  std::string address;

  /**
   * The port of the node we are sending the pages to
   */
  int32_t port;

  /**
   *
   */
  uint64_t numberOfProcessingThreads;

  /**
   *
   */
  uint64_t numberOfNodes;

  /**
   *
   */
  std::pair<uint64_t, std::string> pageSetID;

  /**
   * How many times should we try to connect to the node in case of failure
   */
  uint64_t maxRetries;

  /**
   * The queue where we are getting the pages from
   */
  PDBPageQueuePtr queue;

  /**
   * The logger
   */
  PDBLoggerPtr logger;

  /**
   * The communicator to the node
   */
  PDBCommunicatorPtr comm;
};

}

#endif //PDB_PAGESENDER_H
