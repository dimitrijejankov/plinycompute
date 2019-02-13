#include <cstdint>
#include <string>
#include <HeapRequest.h>


int64_t pdb::RequestFactory::waitForBytes(pdb::PDBLoggerPtr logger, pdb::PDBCommunicatorPtr communicatorPtr,
                                          char *buffer, size_t bufferSize, std::string error) {

  // get the response and process it
  size_t objectSize = communicatorPtr->getSizeOfNextObject();
  if (objectSize == 0) {

    // log the error
    error = "waitForBytes: not able to get next object size";
    logger->error(error);

    // we are done here
    return -1;
  }

  // is the object larger than the buffer
  if(objectSize > bufferSize) {

    // log the error
    error = "waitForBytes: object larger than the buffer";
    logger->error(error);

    return -1;
  }

  // receive bytes
  bool success = communicatorPtr->receiveBytes(buffer, error);

  // did we fail
  if(!success) {
    return -1;
  }

  return objectSize;
}