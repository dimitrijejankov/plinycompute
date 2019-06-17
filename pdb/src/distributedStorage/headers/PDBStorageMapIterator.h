#pragma once

#include <string>

#include "PDBStorageIterator.h"
#include "PDBAggregationResultTest.h"

namespace pdb {


// predefine
template <typename T, typename Enable = void>
class PDBStorageMapIterator;

// enable this if it has both a get key and get value
template <class T>
class PDBStorageMapIterator<T, typename std::enable_if<hasGetKey<T>::value and hasGetValue<T>::value>::type> : public PDBStorageIterator<T> {
public:

  // declare upfront the key and the value types
  using Value = typename std::remove_reference<decltype(std::declval<T>().getValue())>::type;
  using Key = typename std::remove_reference<decltype(std::declval<T>().getKey())>::type;

  PDBStorageMapIterator(std::string address, int port, int maxRetries, std::string set, std::string db) : address(std::move(address)),
                                                                                                          port(port),
                                                                                                          maxRetries(maxRetries),
                                                                                                          set(std::move(set)),
                                                                                                          db(std::move(db)) {

    // init the logger
    logger = std::make_shared<PDBLogger>("setIterator");
  }

  /**
   * Checks if there is another record that we haven't visited
   * @return true if there is false otherwise
   */
  bool hasNextRecord() override {

    // are we starting out
    if (buffer == nullptr && !getNextPage(true)) {
      return false;
    }

    do {

      // if we found an iterator that is not equal to the end return true, since that means that we have one
      if(currIterator != currMap->end()) {
        return true;
      }

      // get the next page since we need it.
    } while (getNextPage(false));

    // we are out of pages
    return false;
  }

  /**
   * Returns the next record.
   * @return returns the record if there is one nullptr otherwise
   */
  pdb::Handle<T> getNextRecord() override {

    // are we starting out
    if (buffer == nullptr && !getNextPage(true)) {
      return nullptr;
    }

    do {

      // do we have a record
      if(currIterator != currMap->end()) {

        // set the return value
        returnValue.getKey() = *((Key*) currIterator.getKeyPtr());
        returnValue.getValue() = *((Value *) currIterator.getValuePtr());

        // increment the iterator
        ++currIterator;

        // IMPORTANT this assumes that the returnValue is not on an allocation block! Otherwise there will be problems
        // since it will do reference counting
        char *location = (char *) &returnValue;
        location -= REF_COUNT_PREAMBLE_SIZE;
        return (RefCountedObject<T> *) location;
      }

      // get the next page since we need it.
    } while (getNextPage(false));

    // we are out of pages
    return nullptr;
  }

private:

  /**
   * Grab the next page
   * @return true if we could grab the next page
   */
  bool getNextPage(bool isFirst) {

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

    // grab the current map
    currMap = ((Record<Map<Key, Value>> *) (buffer.get()))->getRootObject();

    // grab an iterator to the beginning of the map
    currIterator = currMap->begin();

    // we succeeded
    return true;
  }

  /**
   * the address of the manager
   */
  std::string address;

  /**
   * the port of the manager
   */
  int port = -1;

  /**
   * How many times should we retry to connect to the manager if we fail
   */
  int maxRetries = 1;

  /**
   * the logger
   */
  PDBLoggerPtr logger;

  /**
   * The set this iterator belongs to
   */
  std::string set;

  /**
   * The database the set belongs to
   */
  std::string db;

  /**
   * The number of the page we want to get
   */
  uint64_t currPage = 0;

  /**
   * The node we want to grab the page from
   */
  std::string currNode = "none";

  /**
   * The buffer we are storing the records
   */
  std::unique_ptr<char[]> buffer;

  /**
   * The size of the buffer
   */
  size_t bufferSize = 0;

  /**
   * The map we are currently iterating over
   */
  pdb::Handle<pdb::Map<Key, Value>> currMap;

  /**
   * The the current iterator
   */
  PDBMapIterator<Key, Value> currIterator;

  /**
   * This is used to return the value
   */
  T returnValue;
};

// enable this if it does not have the right methods
template <class T>
class PDBStorageMapIterator<T, typename std::enable_if<!hasGetKey<T>::value or !hasGetValue<T>::value>::type> : public PDBStorageIterator<T> {
public:

  PDBStorageMapIterator(std::string, int, int, std::string, std::string) {};

  /**
   * Throws exception since this is not supposed to happen
   */
  bool hasNextRecord() override {
    throw runtime_error(std::string("Type ") + typeid(T).name() + "does not have getKey or getValue.");
  }

  /**
   * Throws exception since this is not supposed to happen
   */
  pdb::Handle<T> getNextRecord() override {
    throw runtime_error(std::string("Type ") + typeid(T).name() + "does not have getKey or getValue.");
  }
};

}