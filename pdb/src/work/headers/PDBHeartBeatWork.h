//
// Created by dimitrije on 10/4/18.
//

#ifndef PDB_PDBHEARTBEATWORK_H
#define PDB_PDBHEARTBEATWORK_H

#include <PDBCatalogClient.h>
#include "PDBWork.h"

class PDBHeartBeatWork;
typedef std::shared_ptr<PDBHeartBeatWork> PDBHeartBeatWorkPtr;

class PDBHeartBeatWork : public pdb::PDBWork {
 public:

  explicit PDBHeartBeatWork(pdb::PDBCatalogClient *server);
  ~PDBHeartBeatWork() = default;

  /**
   * This is where we execute our work
   * @param callerBuzzer - the buzzer linked to this work
   */
  void execute(PDBBuzzerPtr callerBuzzer) override;

  /**
   * We use this method to indicate
   */
  void stop();

 private:

  /**
   * There is going to be a 10 * NODE_PING_DELAY seconds delay between one round of heartbeats
   * And a NODE_PING_DELAY between each of the heartbeats
   */
  const uint32_t NODE_PING_DELAY = 1;

  /**
   * The catalog client we use to grab the info about the nodes in the cluster
   */
  pdb::PDBCatalogClient *client;

  /**
   * Are we still running if not this is true
   */
  atomic_bool isStopped;

  /**
    * Logger to capture debug information about the heart beat
    */
  pdb::PDBLoggerPtr logger;

  /**
   * This will be sending the heartbeats
   * @param address - the address where we want to send
   * @param port - the port where we are sending
   */
  bool sendHeartBeat(const std::string &address, int32_t port);
};

#endif //PDB_PDBHEARTBEATWORK_H
