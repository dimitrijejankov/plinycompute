#include <utility>

#include <PDBVector.h>
#include <PDBCommunicator.h>
#include <HeapRequest.h>
#include <StoGetNextPageRequest.h>
#include <StoGetNextPageResult.h>
#include <snappy.h>

namespace pdb {

template<class T>
PDBStorageVectorIterator<T>::PDBStorageVectorIterator(string address,
                                                      int port,
                                                      int maxRetries,
                                                      string set,
                                                      string db) : address(std::move(address)),
                                                                   port(port),
                                                                   maxRetries(maxRetries),
                                                                   set(std::move(set)),
                                                                   db(std::move(db)) {

  // init the logger
  logger = std::make_shared<PDBLogger>("setIterator");
}

template<class T>
bool PDBStorageVectorIterator<T>::hasNextRecord() {

  // are we starting out
  if (buffer == nullptr && !getNextPage(true)) {
    return false;
  }

  do {

    // grab the vector
    Handle<Vector<Handle<T>>> pageVector = ((Record<Vector<Handle<T>>> *) (buffer.get()))->getRootObject();

    // does this page have more records
    if (currRecord < pageVector->size()) {
      return true;
    }

    // get the next page since we need it.
  } while (getNextPage(false));

  // we are out of pages
  return false;
}

template<class T>
Handle<T> PDBStorageVectorIterator<T>::getNextRecord() {

  // are we starting out
  if (buffer == nullptr && !getNextPage(true)) {
    return nullptr;
  }

  do {

    // grab the vector
    Handle<Vector<Handle<T>>> pageVector = ((Record<Vector<Handle<T>>> *) (buffer.get()))->getRootObject();

    // does this page have more records
    if (currRecord < pageVector->size()) {
      return (*pageVector)[currRecord++];
    }

    // get the next page since we need it.
  } while (getNextPage(false));

  // we are out of pages
  return nullptr;
}

template<class T>
bool PDBStorageVectorIterator<T>::getNextPage(bool isFirst) {

  // the buffer for the compressed data
  std::unique_ptr<char[]> compressedBuffer;
  size_t compressedBufferSize;

  // the communicator
  PDBCommunicatorPtr comm = std::make_shared<PDBCommunicator>();
  string errMsg;

  // try multiple times if we fail to connect
  int numRetries = 0;
  while (numRetries <= maxRetries) {

    // connect to the server
    if (!comm->connectToInternetServer(logger, port, address, errMsg)) {

      // log the error
      logger->error(errMsg);
      logger->error("Can not connect to remote server with port=" + std::to_string(port) + " and address=" + address + ");");

      // throw an exception
      throw std::runtime_error(errMsg);
    }

    // we connected
    break;
  }

  // make a block to send the request
  const UseTemporaryAllocationBlock tempBlock{1024};

  // make the request
  Handle<StoGetNextPageRequest> request = makeObject<StoGetNextPageRequest>(db, set, currPage, currNode, isFirst);

  // send the object
  if (!comm->sendObject(request, errMsg)) {

    // yeah something happened
    logger->error(errMsg);
    logger->error("Not able to send request to server.\n");

    // throw an exception
    throw std::runtime_error(errMsg);
  }

  // get the response and process it
  bool success;
  Handle<StoGetNextPageResult> result = comm->getNextObject<StoGetNextPageResult>(success, errMsg);

  // did we get a response
  if (result == nullptr) {

    // throw an exception
    throw std::runtime_error(errMsg);
  }

  // do we have a next page
  if (!result->hasNext) {
    return false;
  }

  // set the node and the page
  currNode = result->nodeID;
  currPage = result->page + 1;
  compressedBufferSize = result->pageSize;

  // init the compressed buffer
  compressedBuffer = std::unique_ptr<char[]>(new char[result->pageSize]);

  // check if we failed to allocate
  if (compressedBuffer == nullptr) {
    throw std::bad_alloc();
  }

  // we start from the first record
  currRecord = 0;

  // read the size
  auto readSize = RequestFactory::waitForBytes(logger, comm, compressedBuffer.get(), compressedBufferSize, errMsg);

  // did we read anything
  if (readSize == -1) {
    throw std::runtime_error(errMsg);
  }

  // get the uncompressed size
  size_t uncompressedSize = 0;
  snappy::GetUncompressedLength(compressedBuffer.get(), compressedBufferSize, &uncompressedSize);

  // allocate some memory if we need it
  if (bufferSize < uncompressedSize) {

    // allocate the memory
    buffer = std::unique_ptr<char[]>(new char[uncompressedSize]);

    // check if we failed to allocate
    if (buffer == nullptr) {
      throw std::bad_alloc();
    }
  }

  // uncompress and copy to buffer
  snappy::RawUncompress((char *) compressedBuffer.get(), compressedBufferSize, (char *) buffer.get());

  // we succeeded
  return true;
}

}