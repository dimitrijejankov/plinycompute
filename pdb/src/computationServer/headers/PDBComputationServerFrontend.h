//
// Created by dimitrije on 2/19/19.
//

#ifndef PDB_COMPUTATIONSERVER_H
#define PDB_COMPUTATIONSERVER_H

#include "PDBComputationStatsManager.h"
#include <ServerFunctionality.h>
#include <ExJob.h>
#include <mutex>

namespace pdb {

class PDBComputationServerFrontend : public ServerFunctionality {

public:

  void init() override;

  void registerHandlers(PDBServer &forMe) override;

private:

  bool executeJob(pdb::Handle<ExJob> &job);

  bool scheduleJob(PDBCommunicator &temp, pdb::Handle<ExJob> &job, std::string &errMsg);

  bool runScheduledJob(PDBCommunicator &communicator, string &errMsg);

  bool removeUnusedPageSets(const std::vector<pair<uint64_t, std::string>>& pageSets);

  /**
   * This manages the stats about each computation
   */
  PDBComputationStatsManager statsManager;

  /**
   * The logger for this thing
   */
  pdb::PDBLoggerPtr logger;
};


}


#endif //PDB_COMPUTATIONSERVER_H
