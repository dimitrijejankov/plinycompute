#include "PDBConnectionManager.h"

#include <utility>

pdb::PDBConnectionManager::PDBConnectionManager(const pdb::NodeConfigPtr &config, PDBLoggerPtr logger) : logger(std::move(logger)) {
  maxRetries = config->maxRetries;
  listenPort = config->port;
  externalSocket = -1;
}

pdb::PDBConnectionManager::PDBConnectionManager(pdb::PDBLoggerPtr logger) : logger(std::move(logger))  {
  maxRetries = 5;
  listenPort = -1;
  externalSocket = -1;
}

bool pdb::PDBConnectionManager::init() {

  // we are good no need to setup an external socket
  if(listenPort == -1) {
    return true;
  }

  // wait for an internet socket
  std::string errMsg;
  externalSocket = socket(AF_INET, SOCK_STREAM, 0);

  // added by Jia to avoid TimeWait state for old sockets
  int optval = 1;
  if (setsockopt(externalSocket, SOL_SOCKET, SO_REUSEADDR, &optval, sizeof(optval)) < 0) {
    logger->error("PDBServer: couldn't setsockopt");
    logger->error(strerror(errno));
    std::cout << "PDBServer: couldn't setsockopt:" << strerror(errno) << std::endl;
    close(externalSocket);
    externalSocket = -1;
    return false;
  }

  // check for errors
  if (externalSocket < 0) {
    logger->error("PDBServer: could not get FD to internet socket");
    logger->error(strerror(errno));
    close(externalSocket);
    externalSocket = -1;
    return false;
  }

  // bind the socket FD
  struct sockaddr_in serverAddress{};
  bzero((char *) &serverAddress, sizeof(serverAddress));
  serverAddress.sin_family = AF_INET;
  serverAddress.sin_addr.s_addr = INADDR_ANY;
  serverAddress.sin_port = htons((uint16_t) listenPort);
  int retVal = ::bind(externalSocket, (struct sockaddr *) &serverAddress, sizeof(serverAddress));
  if (retVal < 0) {
    logger->error("PDBServer: could not bind to internet socket");
    logger->error(strerror(errno));
    close(externalSocket);
    externalSocket = -1;
    return false;
  }
  logger->trace("PDBServer: about to listen to the Internet for a connection");

  // set the backlog on the socket
  if (::listen(externalSocket, 100) != 0) {
    logger->error("PDBServer: listen error");
    logger->error(strerror(errno));
    close(externalSocket);
    externalSocket = -1;
    return false;
  }

  // load that we are ready
  logger->trace("PDBServer: ready to go!");

  // we initialized the socket!
  return true;
}

pdb::PDBCommunicatorPtr pdb::PDBConnectionManager::connectToInternetServer(const pdb::PDBLoggerPtr& logToMe,
                                                                           int portNumber, const std::string& serverAddress,
                                                                           std::string &errMsg) {

  //
  logToMe->trace("PDBCommunicator: About to connect to the remote host");

  struct addrinfo hints{};
  struct addrinfo *result, *rp;
  char port[10];
  sprintf(port, "%d", portNumber);

  memset(&hints, 0, sizeof(struct addrinfo));
  hints.ai_family = AF_INET;
  hints.ai_socktype = SOCK_STREAM;
  hints.ai_flags = 0;
  hints.ai_protocol = 0;

  int s = getaddrinfo(serverAddress.c_str(), port, &hints, &result);
  if (s != 0) {
    logToMe->error("PDBCommunicator: could not get addr info");
    logToMe->error(strerror(errno));
    errMsg = "Could not get addr info ";
    errMsg += strerror(errno);
    std::cout << errMsg << std::endl;
    return nullptr;
  }

  bool connected = false;
  int32_t socketFD = -1;
  for (rp = result; rp != nullptr; rp = rp->ai_next) {
    int count = 0;
    while (count <= maxRetries) {
      logToMe->trace("PDBCommunicator: creating socket....");
      socketFD = socket(rp->ai_family, rp->ai_socktype, rp->ai_protocol);
      if (socketFD == -1) {
        continue;
      }
      if (::connect(socketFD, rp->ai_addr, rp->ai_addrlen) != -1) {
        connected = true;
        break;
      }
      count++;
      std::cout << "Connection error, to retry..." << std::endl;
      sleep(1);
      close(socketFD);
      socketFD = -1;
    }
    if (connected) {
      break;
    }
  }

  // check for error
  if (rp == nullptr) {
    logToMe->error("PDBCommunicator: could not connect to server: address info is null");
    logToMe->error(strerror(errno));
    errMsg = "Could not connect to server: address info is null with ip=" + serverAddress +
        ", and port=" + port;
    errMsg += strerror(errno);
    std::cout << errMsg << std::endl;
    return nullptr;
  }
  freeaddrinfo(result);

  // create the communicator
  auto comm = std::make_shared<PDBCommunicator>();
  comm->logToMe = logToMe;
  comm->portNumber = portNumber;
  comm->serverAddress = serverAddress;
  comm->needToSendDisconnectMsg = true;
  comm->socketFD = socketFD;

  logToMe->trace("PDBCommunicator: Successfully connected to the remote host");
  logToMe->trace("PDBCommunicator: Socket FD is " + std::to_string(socketFD));

  // return the communicator
  return std::move(comm);
}

pdb::PDBCommunicatorPtr pdb::PDBConnectionManager::listen(std::string &errMsg) {

  // if there is no port set we need to listen to jump out of it
  if(listenPort == -1 || externalSocket == -1) {
    throw std::runtime_error("Can not listen to a closed socket.");
  }

  struct sockaddr_in cli_addr{};
  socklen_t clilen = sizeof(cli_addr);
  bzero((char*)&cli_addr, sizeof(cli_addr));
  logger->info("PDBCommunicator: about to wait for request from Internet");

  // wait for a connection
  auto socketFD = accept(externalSocket, (struct sockaddr*)&cli_addr, &clilen);
  if (socketFD < 0) {
    logger->error("PDBCommunicator: could not get FD to internet socket");
    logger->error(strerror(errno));
    errMsg = "Could not get socket ";
    errMsg += strerror(errno);
    close(socketFD);
    return nullptr;
  }
  logger->info("PDBCommunicator: got request from Internet");

  // make the communicator
  auto comm = std::make_shared<PDBCommunicator>();
  comm->socketClosed = false;
  comm->socketFD = socketFD;
  comm->logToMe = logger;

  // return the communicator
  return std::move(comm);
}

const pdb::PDBLoggerPtr &pdb::PDBConnectionManager::getLogger() const {
  return logger;
}
