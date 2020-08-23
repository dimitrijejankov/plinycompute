#include "PDBCommunicator.h"
#include "NodeConfig.h"

namespace pdb {

class PDBConnectionManager {
public:

  PDBConnectionManager(const NodeConfigPtr &config, PDBLoggerPtr logger);

  explicit PDBConnectionManager(PDBLoggerPtr  logger);

  // initializes the external socket
  bool init();

  // listens to the external socket, if it succeeds returns the communicator otherwise returns null
  PDBCommunicatorPtr pointToInternet(std::string &errMsg);

  // this connects to a server
  PDBCommunicatorPtr connectToInternetServer(const PDBLoggerPtr &logToMe, int portNumber,
                                             const std::string &serverAddress, std::string &errMsg);

private:

  // listen to this port
  int32_t listenPort;

  // the number of retries
  int32_t maxRetries;

  // the socket to us
  int32_t externalSocket;

  // the logger
  PDBLoggerPtr logger;
};

//
using PDBConnectionManagerPtr = std::shared_ptr<PDBConnectionManager>;
}