#include <DisRemoveSet.h>
#include "PDBDistributedStorageClient.h"

namespace pdb {

bool PDBDistributedStorageClient::clearSet(const string &dbName, const string &setName, std::string &errMsg) {
  return RequestFactory::heapRequest<DisClearSet, SimpleRequestResult, bool>(
      logger, port, address, false, 1024,
      [&](Handle<SimpleRequestResult> result) {

        // check if the result is correct
        if (result != nullptr && result->getRes().first) {
          return true;
        }

        // log the error
        errMsg = "Error clearing set: ";
        errMsg += result != nullptr ? result->getRes().second : "Bad response";
        logger->error(errMsg);

        return false;
      },
      dbName, setName);
}

bool PDBDistributedStorageClient::removeSet(const string &dbName, const string &setName, std::string &errMsg) {
  return RequestFactory::heapRequest<DisRemoveSet, SimpleRequestResult, bool>(
      logger, port, address, false, 1024,
      [&](Handle<SimpleRequestResult> result) {

        // check if the result is correct
        if (result != nullptr && result->getRes().first) {
          return true;
        }

        // log the error
        errMsg = "Error removing set: ";
        errMsg += result != nullptr ? result->getRes().second : "Bad response";
        logger->error(errMsg);

        return false;
      },
      dbName, setName);
}

}
