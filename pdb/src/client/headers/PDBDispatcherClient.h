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

#ifndef OBJECTQUERYMODEL_DISPATCHERCLIENT_H
#define OBJECTQUERYMODEL_DISPATCHERCLIENT_H

#include "ServerFunctionality.h"
#include "Handle.h"
#include "PDBVector.h"
#include "PDBObject.h"
#include "PDBCatalogClient.h"

namespace pdb {

/**
 * this class serves as a dispatcher client to talk with the DispatcherServer
 * to send Vector<Objects> from clients to the distributed storage server.
 */
class PDBDispatcherClient : public ServerFunctionality {

public:

  PDBDispatcherClient() = default;

  /**
   * Constructor for the client
   * @param portIn - the port of the manager
   * @param addressIn - the address of the manager
   * @param myLoggerIn - the logger of the client
   */
  PDBDispatcherClient(int portIn, std::string addressIn, PDBLoggerPtr myLoggerIn)
                      : port(portIn), address(std::move(addressIn)), logger(std::move(myLoggerIn)) {};

  ~PDBDispatcherClient() = default;

  /**
   * Registers the handles needed for the server functionality
   * @param forMe
   */
  void registerHandlers(PDBServer &forMe) override {};

  /**
   * Send the data to the dispatcher
   * @param setAndDatabase - the set and database pair where we want to
   * @return true if we succeed false otherwise
   */
  template<class DataType>
  bool sendData(const std::string &db, const std::string &set, Handle<Vector<Handle<DataType>>> dataToSend, std::string &errMsg);

private:

  /**
   * The port of the manager
   */
  int port = -1;

  /**
   * The address of the manager
   */
  std::string address;

  /**
   * The logger of the client
   */
  PDBLoggerPtr logger;
};

}

#include "PDBDispatcherClientTemplate.cc"

#endif  // OBJECTQUERYMODEL_DISPATCHERCLIENT_H
