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

#ifndef OBJECTQUERYMODEL_DISPATCHER_H
#define OBJECTQUERYMODEL_DISPATCHER_H

#include "ServerFunctionality.h"
#include "PDBLogger.h"
#include "PDBWork.h"
#include "UseTemporaryAllocationBlock.h"
#include "PDBVector.h"

#include "NodeDispatcherData.h"
#include "StorageClient.h"
#include "PDBDispatcherPolicy.h"

#include <string>
#include <queue>
#include <unordered_map>
#include <vector>

namespace pdb {

/**
 * The DispatcherServer partitions and then forwards a Vector of pdb::Objects received from a
 * PDBDispatcherClient to the proper storage servers
 */
class PDBDispatcherServer : public ServerFunctionality {

public:

    PDBDispatcherServer() = default;

    ~PDBDispatcherServer() = default;

    void initialize();

    /**
     * Inherited function from ServerFunctionality
     * @param forMe
     */
    void registerHandlers(PDBServer& forMe) override;

private:

  PDBDispatcherPolicyPtr policy;

  PDBLoggerPtr logger;
};
}


#endif  // OBJECTQUERYMODEL_DISPATCHER_H
