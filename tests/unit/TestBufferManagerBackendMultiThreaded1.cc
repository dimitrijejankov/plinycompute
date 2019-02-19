//
// Created by dimitrije on 2/3/19.
//

#include <PDBBufferManagerBackEnd.h>
#include <GenericWork.h>
#include "TestBufferManagerBackend.h"

namespace pdb {

// this test checks whether anonymous pages work on the backend
TEST(BufferManagerBackendTest, Test3) {

  // parameters
  const int numPages = 4000;
  const int numThreads = 4;
  const int maxPageSize = 64;

  // the page sizes we are testing
  std::vector<size_t> pageSizes {8, 16, 32};

  int64_t curPage = 0;

  vector<bool> pinned(numPages, false);
  vector<bool> frozen(numPages, false);
  std::unordered_map<int64_t, int64_t> pages;

  // allocate memory
  std::unique_ptr<char[]> memory(new char[numPages * maxPageSize]);

  // make the shared memory object
  PDBSharedMemory sharedMemory{};
  sharedMemory.pageSize = maxPageSize;
  sharedMemory.numPages = numPages;
  sharedMemory.memory = memory.get();

  // create the buffer manager
  pdb::PDBBufferManagerBackEnd<MockRequestFactory> bufferManager(sharedMemory);

  MockRequestFactory::_requestFactory = std::make_shared<MockRequestFactoryImpl>();

  // to sync the threads
  std::mutex m;

  /// 0. Init the mock server

  MockServer server;
  ON_CALL(server, getConfiguration()).WillByDefault(testing::Invoke(
      [&]() {
        return std::make_shared<pdb::NodeConfig>();
      }));

  EXPECT_CALL(server, getConfiguration).Times(testing::AtLeast(1));

  bufferManager.recordServer(server);

  /// 1. Mock the anonymous pages request

  ON_CALL(*MockRequestFactory::_requestFactory, getAnonPage).WillByDefault(testing::Invoke(
      [&](pdb::PDBLoggerPtr &myLogger, int port, const std::string &address, pdb::PDBPageHandle onErr,
          size_t bytesForRequest, const std::function<pdb::PDBPageHandle(pdb::Handle<pdb::BufGetPageResult>)> &processResponse, size_t minSize) {

        // do the book keeping
        int64_t myPage;
        {
          unique_lock<std::mutex> lck(m);

          // increment the current page
          myPage = curPage++;

          // mark it as pinned
          pinned[myPage] = true;
        }

        const pdb::UseTemporaryAllocationBlock tempBlock{1024};

        // make the page
        pdb::Handle<pdb::BufGetPageResult> returnPageRequest = pdb::makeObject<pdb::BufGetPageResult>(myPage * maxPageSize, myPage, true, false, -1, minSize, "", "");

        // return true since we assume this succeeded
        return processResponse(returnPageRequest);
      }
  ));

  // it should call send object exactly 98 times
  EXPECT_CALL(*MockRequestFactory::_requestFactory, getAnonPage).Times(testing::AtLeast(1));

  /// 2. Mock the unpin page

  ON_CALL(*MockRequestFactory::_requestFactory, unpinPage).WillByDefault(testing::Invoke(
      [&](pdb::PDBLoggerPtr &myLogger, int port, const std::string &address, bool onErr, size_t bytesForRequest,
          const std::function<bool(pdb::Handle<pdb::SimpleRequestResult>)> &processResponse, PDBSetPtr &set, size_t pageNum, bool isDirty) {

        // do the book keeping
        {
          unique_lock<std::mutex> lck(m);

          // expect it to be pinned when you return it
          EXPECT_TRUE(pinned[pageNum]);

          // mark it as unpinned
          pinned[pageNum] = false;
        }

        const pdb::UseTemporaryAllocationBlock tempBlock{1024};

        // check the page
        EXPECT_GE(pageNum, 0);
        EXPECT_LE(pageNum, numPages);

        // make the page
        pdb::Handle<pdb::SimpleRequestResult> returnPageRequest = pdb::makeObject<pdb::SimpleRequestResult>(true, "");

        // return true since we assume this succeeded
        return processResponse(returnPageRequest);
      }
  ));

  // it should call send object exactly twice
  EXPECT_CALL(*MockRequestFactory::_requestFactory, unpinPage).Times(testing::AtLeast(1));

  /// 3. Mock the freeze size

  ON_CALL(*MockRequestFactory::_requestFactory, freezeSize).WillByDefault(testing::Invoke(
      [&](pdb::PDBLoggerPtr &myLogger, int port, const std::string address, bool onErr,
          size_t bytesForRequest, const std::function<bool(pdb::Handle<pdb::BufFreezeRequestResult>)> &processResponse,
          pdb::PDBSetPtr setPtr, size_t pageNum, size_t numBytes) {

        // do the book keeping
        {
          unique_lock<std::mutex> lck(m);

          // expect not to be frozen
          EXPECT_FALSE(frozen[pageNum]);

          // mark it as frozen
          frozen[pageNum] = true;
        }

        const pdb::UseTemporaryAllocationBlock tempBlock{1024};

        // check the page
        EXPECT_GE(pageNum, 0);
        EXPECT_LE(pageNum, numPages);

        // make the page
        pdb::Handle<pdb::BufFreezeRequestResult> returnPageRequest = pdb::makeObject<pdb::BufFreezeRequestResult>(true);

        // return true since we assume this succeeded
        return processResponse(returnPageRequest);
      }
  ));

  // it should call send object exactly once
  EXPECT_CALL(*MockRequestFactory::_requestFactory, freezeSize).Times(0);

  /// 4. Mock the pin page

  ON_CALL(*MockRequestFactory::_requestFactory, pinPage).WillByDefault(testing::Invoke(
      [&](pdb::PDBLoggerPtr &myLogger, int port, const std::string &address, bool onErr,
          size_t bytesForRequest, const std::function<bool(pdb::Handle<pdb::BufPinPageResult>)> &processResponse,
          const pdb::PDBSetPtr &setPtr, size_t pageNum) {

        // do the book keeping
        {
          unique_lock<std::mutex> lck(m);

          // mark it as unpinned
          pinned[pageNum] = true;
        }

        const pdb::UseTemporaryAllocationBlock tempBlock{1024};

        // check the page
        EXPECT_GE(pageNum, 0);
        EXPECT_LE(pageNum, numPages);

        // make the page
        pdb::Handle<pdb::BufPinPageResult> returnPageRequest = pdb::makeObject<pdb::BufPinPageResult>(pageNum * maxPageSize, true);

        // return true since we assume this succeeded
        return processResponse(returnPageRequest);
      }
  ));

  // it should call send object exactly twice
  EXPECT_CALL(*MockRequestFactory::_requestFactory, pinPage).Times(testing::AtLeast(1));

  /// 5. Mock return anon page

  ON_CALL(*MockRequestFactory::_requestFactory, returnAnonPage).WillByDefault(testing::Invoke(
      [&](pdb::PDBLoggerPtr &myLogger, int port, const std::string &address, bool onErr, size_t bytesForRequest,
          const std::function<bool(pdb::Handle<pdb::SimpleRequestResult>)> &processResponse,
          size_t pageNum, bool isDirty) {

        // do the book keeping
        {
          unique_lock<std::mutex> lck(m);

          // expect it to be pinned when you return it
          ///EXPECT_TRUE(pinned[pageNum]);

          // mark it as unpinned
          pinned[pageNum] = false;
        }

        const pdb::UseTemporaryAllocationBlock tempBlock{1024};

        // check the page
        EXPECT_GE(pageNum, 0);
        EXPECT_LE(pageNum, numPages);

        // make the page
        pdb::Handle<pdb::SimpleRequestResult> returnPageRequest = pdb::makeObject<pdb::SimpleRequestResult>(true, "");

        // return true since we assume this succeeded
        return processResponse(returnPageRequest);
      }
  ));

  // it should call send object exactly 98 times
  EXPECT_CALL(*MockRequestFactory::_requestFactory, returnAnonPage).Times(testing::AtLeast(1));

  {
    // used to sync
    std::atomic<std::int32_t> sync;
    sync = 0;

    // init the worker threads of this server
    auto workers = make_shared<PDBWorkerQueue>(make_shared<PDBLogger>("worker.log"), numThreads + 2);

    // create the buzzer
    atomic_int counter;
    counter = 0;
    PDBBuzzerPtr tempBuzzer = make_shared<PDBBuzzer>([&](PDBAlarm myAlarm, atomic_int &cnt) {
      cnt++;
    });

    // run multiple threads
    for (int t = 0; t < numThreads; ++t) {

      // grab a worker
      PDBWorkerPtr myWorker = workers->getWorker();

      // the thread
      int thread = t;

      // start the thread
      PDBWorkPtr myWork = make_shared<pdb::GenericWork>([&, thread](PDBBuzzerPtr callerBuzzer) {

        // sync the threads to make sure there is more overlapping
        sync++;
        while (sync != numThreads) {}

        int offset = 0;

        std::vector<PDBPageHandle> pageHandles;

        // grab anon pages
        for (int i = 0; i < numPages / numThreads; ++i) {

          // use a different page size
          size_t pageSize = pageSizes[i % 3];

          // grab the page
          auto page = bufferManager.getPage(pageSize);

          // grab the page and fill it in
          char *bytes = (char *) page->getBytes();
          for (char j = 0; j < pageSize; ++j) {
            bytes[j] = static_cast<char>((j + offset + thread) % 128);
          }

          // check
          EXPECT_NE(bytes, nullptr);

          // store page
          pageHandles.push_back(page);

          // unpin the page
          page->unpin();
          EXPECT_EQ(page->getBytes(), nullptr);

          // increment the offset
          offset++;
        }

        // sync the threads to make sure there is more overlapping
        sync++;
        while (sync != 2 * numThreads) {}

        offset = 0;
        for (int i = 0; i < numPages / numThreads; ++i) {

          // use a different page size
          size_t pageSize = pageSizes[i % 3];

          // grab the page
          auto page = pageHandles[i];

          // repin the page
          page->repin();
          EXPECT_NE(page->getBytes(), nullptr);

          // grab the page and fill it in
          char *bytes = (char *) page->getBytes();
          for (char j = 0; j < pageSize; ++j) {
            EXPECT_EQ(bytes[j], static_cast<char>((j + offset + thread) % 128));
          }

          // unpin the page
          page->unpin();
          EXPECT_EQ(page->getBytes(), nullptr);

          // increment the offset
          offset++;
        }

        // remove all the page handles
        pageHandles.clear();

        // excellent everything worked just as expected
        callerBuzzer->buzz(PDBAlarm::WorkAllDone, counter);
      });

      // run the work
      myWorker->execute(myWork, tempBuzzer);
    }

    // wait until all the nodes are finished
    while (counter < numThreads) {
      tempBuzzer->wait();
    }
  }

  // just to remove the mock object
  MockRequestFactory::_requestFactory = nullptr;
  bufferManager.parent = nullptr;
}

}