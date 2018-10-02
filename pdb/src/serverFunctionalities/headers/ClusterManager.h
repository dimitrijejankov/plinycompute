/*****************************************************************************
 *                                                                           *
 *  Copyright 2018 Rice University                                           *
 *                                                                           *
 *  Licensed under the Apache License, Version 2.0 (the "License");          *
 *  you may not use this file except in compliance with the License.         *
 *  You may obtain a copy of the License at                                  *
 *                                                                           *
 *      http://www.apache.org/licenses/LICENSE-2.0                           *
 *                                                                           *
 *  Unless required by applicable law or agreed to in writing, software      *
 *  distributed under the License is distributed on an "AS IS" BASIS,        *
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. *
 *  See the License for the specific language governing permissions and      *
 *  limitations under the License.                                           *
 *                                                                           *
 *****************************************************************************/

#ifndef PDB_CLUSTERMANAGER_H
#define PDB_CLUSTERMANAGER_H

#include <mutex>
#include <thread>
#include <sys/sysinfo.h>
#include "ServerFunctionality.h"

namespace pdb {

class ClusterManager : public ServerFunctionality {
public:

  ClusterManager(std::string address, int32_t port, bool isManager);


  void registerHandlers(PDBServer& forMe) override;


  bool syncCluster(const std::string &managerAddress, int managerPort, std::string &error);

private:

  /**
   * Logger for the cluster manager
   */
  PDBLoggerPtr logger;

  /**
   * A mutex to sync the cluster
   */
  std::mutex serverMutex;

  /**
   * The ip of the node
   */
  std::string address;

  /**
   * The port of the node
   */
  int32_t port;

  /**
   * True if this node is the manager
   */
  bool isManager;

  /**
   * The size of the memory on this machine
   */
  int64_t totalMemory = -1;

  /**
   * The number of cores on this machine
   */
  int32_t numCores = -1;
};

}


#endif //PDB_CLUSTERMANAGER_H
