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


#ifndef PDB_COMMUN_C
#define PDB_COMMUN_C

#include "PDBDebug.h"
#include "BuiltInObjectTypeIDs.h"
#include "Handle.h"
#include <iostream>
#include <utility>
#include <netdb.h>
#include <netinet/in.h>
#include "Object.h"
#include "PDBVector.h"
#include "CloseConnection.h"
#include "UseTemporaryAllocationBlock.h"
#include "InterfaceFunctions.h"
#include "PDBCommunicator.h"
#include <stdio.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <sys/types.h>
#include <sys/uio.h>
#include <sys/un.h>
#include <netinet/in.h>
#include <unistd.h>


#define MAX_RETRIES 5


namespace pdb {

PDBCommunicator::PDBCommunicator() {
    readCurMsgSize = false;
    socketFD = -1;
    nextTypeID = NoMsg_TYPEID;
    socketClosed = true;
    // Jia: moved this logic from Chris' message-based communication framework to here
    needToSendDisconnectMsg = false;
}

void PDBCommunicator::setNeedsToDisconnect(bool option) {
    needToSendDisconnectMsg = option;
}

PDBCommunicator::~PDBCommunicator() {

// Jia: moved below logic from Chris' message-based communication to here.
// tell the server that we are disconnecting (note that needToSendDisconnectMsg is
// set to true only if we are a client and we want to close a connection to the server
#ifdef __APPLE__
    if (needToSendDisconnectMsg && socketFD > 0) {
        const UseTemporaryAllocationBlock tempBlock{1024};
        Handle<CloseConnection> temp = makeObject<CloseConnection>();
        logToMe->trace("PDBCommunicator: closing connection to the server");
        std::string errMsg;
        if (!sendObject(temp, errMsg)) {
            logToMe->trace("PDBCommunicator: could not send close connection message");
        }
    }

    if (socketFD >= 0) {
        close(socketFD);
        socketClosed = true;
        socketFD = -1;
    }
#else


    if (needToSendDisconnectMsg && socketFD >= 0) {
        close(socketFD);
        socketFD = -1;
    } else if (!needToSendDisconnectMsg && socketFD >= 0) {
        shutdown(socketFD, SHUT_WR);
        // below logic doesn't work!
        /*
        char c;
        ssize_t res = recv(socketFD, &c, 1, MSG_PEEK);
        if (res == 0) {
            std :: cout << "server socket closed" << std :: endl;
        } else {
            std :: cout << "there is some error in the socket" << std :: endl;
        }
        */
        close(socketFD);
        socketFD = -1;
    }
    socketClosed = true;
#endif
}

int PDBCommunicator::getSocketFD() {
    return socketFD;
}

int16_t PDBCommunicator::getObjectTypeID() {

    if (!readCurMsgSize) {
        getSizeOfNextObject();
    }
    return nextTypeID;
}

size_t PDBCommunicator::getSizeOfNextObject() {

    // if we have previously gotten the size, just return it
    if (readCurMsgSize) {
        logToMe->debug("getSizeOfNextObject: we've done this before");
        return msgSize;
    }

    // make sure we got enough bytes... if we did not, then error out
    // JIANOTE: we may not receive all the bytes at once, so we need a loop
    int receivedBytes = 0;
    int receivedTotal = 0;
    int bytesToReceive = (int)(sizeof(int16_t));
    int retries = 0;
    while (receivedTotal < (int)(sizeof(int16_t))) {
        if ((receivedBytes = read(socketFD,
                                  (char*)((char*)(&nextTypeID) + receivedTotal * sizeof(char)),
                                  bytesToReceive)) < 0) {
            std::string errMsg =
                std::string("PDBCommunicator: could not read next message type") + strerror(errno);
            logToMe->error(errMsg);
            PDB_COUT << errMsg << std::endl;
            nextTypeID = NoMsg_TYPEID;
            msgSize = 0;
            close(socketFD);
            socketFD = -1;
            socketClosed = true;
            return 0;
        } else if (receivedBytes == 0) {
            logToMe->info(
                "PDBCommunicator: the other side closed the socket when we try to read the type");
            nextTypeID = NoMsg_TYPEID;
            PDB_COUT
                << "PDBCommunicator: the other side closed the socket when we try to get next type"
                << std::endl;

            // if (retries < MAX_RETRIES) {
            if (retries < 0) {
                retries++;
                logToMe->info("PDBCommunicator: Retry to see whether network can recover");
                PDB_COUT << "PDBCommunicator: Retry to see whether network can recover"
                         << std::endl;
                continue;
            } else {
                close(socketFD);
                socketFD = -1;
                socketClosed = true;
                msgSize = 0;
                return 0;
            }

        } else {
            logToMe->info(std::string("PDBCommunicator: receivedBytes for reading type is ") +
                          std::to_string(receivedBytes));
            receivedTotal = receivedTotal + receivedBytes;
            bytesToReceive = sizeof(int16_t) - receivedTotal;
        }
    }
    // now we get enough bytes
    logToMe->trace("PDBCommunicator: typeID of next object is " + std::to_string(nextTypeID));
    logToMe->trace("PDBCommunicator: getting the size of the next object:");

    // make sure we got enough bytes... if we did not, then error out
    receivedBytes = 0;
    receivedTotal = 0;
    bytesToReceive = (int)(sizeof(size_t));
    retries = 0;
    while (receivedTotal < (int)(sizeof(size_t))) {
        if ((receivedBytes = read(socketFD,
                                  (char*)((char*)(&msgSize) + receivedTotal * sizeof(char)),
                                  bytesToReceive)) < 0) {
            std::string errMsg = "PDBCommunicator: could not read next message size:" +
                std::to_string(receivedTotal) + strerror(errno);
            logToMe->error(errMsg);
            PDB_COUT << errMsg << std::endl;
            close(socketFD);
            socketFD = -1;

            socketClosed = true;
            msgSize = 0;
            return 0;
        } else if (receivedBytes == 0) {
            logToMe->info(
                "PDBCommunicator: the other side closed the socket when we try to get next size");
            nextTypeID = NoMsg_TYPEID;
            PDB_COUT
                << "PDBCommunicator: the other side closed the socket when we try to get next size"
                << std::endl;
            // if (retries < MAX_RETRIES) {
            if (retries < 0) {
                retries++;
                PDB_COUT << "PDBCommunicator: Retry to see whether network can recover"
                         << std::endl;
                logToMe->info("PDBCommunicator: Retry to see whether network can recover");
                continue;
            } else {
                close(socketFD);
                socketFD = -1;
                socketClosed = true;
                msgSize = 0;
                return 0;
            }

        } else {
            logToMe->info(std::string("PDBCommunicator: receivedBytes for reading size is ") +
                          std::to_string(receivedBytes));
            receivedTotal = receivedTotal + receivedBytes;
            bytesToReceive = sizeof(size_t) - receivedTotal;
        }
    }
    // OK, we did get enough bytes
    logToMe->trace("PDBCommunicator: size of next object is " + std::to_string(msgSize));
    readCurMsgSize = true;
    return msgSize;
}

bool PDBCommunicator::doTheWrite(char* start, char* end) {

    int retries = 0;
    // and do the write
    while (end != start) {

        // write some bytes
        ssize_t numBytes = write(socketFD, start, end - start);
        // make sure they went through
        if (numBytes < 0) {
            logToMe->error("PDBCommunicator: error in socket write");
            logToMe->trace("PDBCommunicator: tried to write " + std::to_string(end - start) +
                           " bytes.\n");
            logToMe->trace("PDBCommunicator: Socket FD is " + std::to_string(socketFD));
            logToMe->error(strerror(errno));
            // if (retries < MAX_RETRIES) {
            if (retries < 0) {
                retries++;
                PDB_COUT << "PDBCommunicator: Retry to see whether network can recover"
                         << std::endl;
                logToMe->info("PDBCommunicator: Retry to see whether network can recover");
                continue;
            } else {
                // std :: cout << "############################################" << std :: endl;
                // std :: cout << "WARNING: CONNECTION CLOSED DUE TO WRITE ERROR AFTER RETRY" << std
                // :: endl;
                // std :: cout << "############################################" << std :: endl;
                close(socketFD);
                socketFD = -1;
                socketClosed = true;
                return false;
            }
        } else {
            logToMe->trace("PDBCommunicator: wrote " + std::to_string(numBytes) + " and are " +
                           std::to_string(end - start - numBytes) + " to go!");
            start += numBytes;
        }
    }
    return true;
}

bool PDBCommunicator::doTheRead(char* dataIn) {

    if (!readCurMsgSize) {
        getSizeOfNextObject();
    }
    readCurMsgSize = false;

    // now, read the rest of the bytes
    char* start = dataIn;
    char* cur = start;

    int retries = 0;
    while (cur - start < (long)msgSize) {

        ssize_t numBytes = read(socketFD, cur, msgSize - (cur - start));
        this->logToMe->trace("PDBCommunicator: received bytes: " + std::to_string(numBytes));

        if (numBytes < 0) {
            logToMe->error(
                "PDBCommunicator: error reading socket when trying to accept text message");
            logToMe->error(strerror(errno));
            close(socketFD);
            socketFD = -1;
            socketClosed = true;
            return false;
        } else if (numBytes == 0) {
            logToMe->info("PDBCommunicator: the other side closed the socket when we do the read");
            PDB_COUT << "PDBCommunicator: the other side closed the socket when we doTheRead"
                     << std::endl;
            // if (retries < MAX_RETRIES) {
            if (retries < 0) {
                retries++;
                logToMe->info("PDBCommunicator: Retry to see whether network can recover");
                PDB_COUT << "PDBCommunicator: Retry to see whether network can recover"
                         << std::endl;
                continue;
            } else {
                close(socketFD);
                socketFD = -1;
                socketClosed = true;
                return false;
            }
        } else {
            cur += numBytes;
        }
        this->logToMe->trace("PDBCommunicator: " + std::to_string(msgSize - (cur - start)) +
                             " bytes to go!");
    }
    return true;
}

bool PDBCommunicator::skipBytes(std::string &errMsg) {

    // if we have previously gotten the size, just return it
    if (!readCurMsgSize) {
        getSizeOfNextObject();
    }

    if (!skipTheRead()) {
        errMsg = "Could not read the next object coming over the wire";
        readCurMsgSize = false;
        return false;
    }

    return true;
}

bool PDBCommunicator::skipTheRead() {

    // make sure the size we got is the most recent one
    if (!readCurMsgSize) {
        getSizeOfNextObject();
    }
    readCurMsgSize = false;

    // the bytes are read in chunks of 1MB
    std::unique_ptr<char[]> memory(new char[1024 * 1024]);

    size_t cur = 0;

    int retries = 0;
    while (cur < (long) msgSize) {

        ssize_t numBytes = read(socketFD, memory.get(), std::min<size_t>(msgSize - cur, 1024 * 1024));
        this->logToMe->trace("PDBCommunicator: received bytes: " + std::to_string(numBytes));

        if (numBytes < 0) {

            // log the error
            logToMe->error("PDBCommunicator: error reading socket when trying to accept text message");
            logToMe->error(strerror(errno));

            // close the connection
            close(socketFD);
            socketFD = -1;
            socketClosed = true;

            // finish
            return false;

        } else if (numBytes == 0) {

            // log the info
            logToMe->info("PDBCommunicator: the other side closed the socket when we do the read");

            // are we out of retries
            if (retries < 0) {

                // retry
                retries++;
                logToMe->info("PDBCommunicator: Retry to see whether network can recover");
                continue;

            } else {

                // close connection
                close(socketFD);
                socketFD = -1;
                socketClosed = true;

                // finish
                return false;
            }
        } else {

            // increment the byte count
            cur += numBytes;
        }
        this->logToMe->trace("PDBCommunicator: " + std::to_string(msgSize - cur) +" bytes to go!");
    }
    return true;
}

// JiaNote: add following functions to enable a stable long connection:
bool PDBCommunicator::isSocketClosed() {

  int error = 0;
  socklen_t len = sizeof (error);
  int retval = getsockopt (socketFD, SOL_SOCKET, SO_ERROR, &error, &len);

  // check the return value
  if (retval != 0) {

    // there was a problem getting the error code
    this->logToMe->trace(std::string("error getting socket error code:") + strerror(retval) + "\n");

    // mark the socket as closed
    socketClosed = true;
    return socketClosed;
  }

  // check if we got an error
  if (error != 0) {

    // log the error
    this->logToMe->trace(std::string("error getting socket error code:") + strerror(retval) + "\n");

    // mark the socket as closed
    socketClosed = true;
    return socketClosed;
  }

  // return the state of the socket
  return socketClosed;
}

}

#endif
