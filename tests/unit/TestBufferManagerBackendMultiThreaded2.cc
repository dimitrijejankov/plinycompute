//
// Created by dimitrije on 2/4/19.
//

#include <PDBBufferManagerBackEnd.h>
#include <GenericWork.h>
#include <random>
#include "TestBufferManagerBackend.h"

namespace pdb {


TEST(BufferManagerBackendTest, Test4) {

  const int maxPageSize = 128;
  const int numRequestsPerPage = 2000;
  const int numPages = 60;

  // note the number of threads must be less than 8 or equal to 8 or else we can exceed the page size
  const int numThreads = 4;

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
  pdb::PDBBufferManagerBackEnd<MockRequestFactory> myMgr(sharedMemory);

  MockRequestFactory::_requestFactory = std::make_shared<MockRequestFactoryImpl>();

  // to sync the threads
  std::mutex m;

  /// 0. Init the mock server

  MockServer server;
  ON_CALL(server, getConfiguration).WillByDefault(testing::Invoke(
      [&]() {
        return std::make_shared<pdb::NodeConfig>();
      }));

  EXPECT_CALL(server, getConfiguration).Times(testing::AtLeast(1));

  myMgr.recordServer(server);

  /// 1. Mock the get page for the set

  ON_CALL(*MockRequestFactory::_requestFactory, getPage).WillByDefault(testing::Invoke(
      [&](pdb::PDBLoggerPtr &myLogger, int port, const std::string &address,pdb::PDBPageHandle onErr,
          size_t bytesForRequest, const std::function<pdb::PDBPageHandle(pdb::Handle<pdb::BufGetPageResult>)> &processResponse,
          const pdb::PDBSetPtr &set, uint64_t pageNum) {

        {
          unique_lock<std::mutex> lock(m);

          // mark it as pinned
          pinned[pageNum] = true;
        }

        const pdb::UseTemporaryAllocationBlock tempBlock{1024};

        // check the page
        EXPECT_GE(pageNum, 0);
        EXPECT_LE(pageNum, numPages);
        EXPECT_TRUE(set->getSetName() == "set1");
        EXPECT_TRUE(set->getDBName() == "DB");

        // make the page
        pdb::Handle<pdb::BufGetPageResult> returnPageRequest = pdb::makeObject<pdb::BufGetPageResult>(pageNum * maxPageSize, pageNum, false, false, -1, maxPageSize, set->getSetName(), set->getDBName());

        // return true since we assume this succeeded
        return processResponse(returnPageRequest);
      }
  ));

  // it should call send object exactly 98 times
  EXPECT_CALL(*MockRequestFactory::_requestFactory, getPage).Times(testing::AtLeast(1));


  /// 2. Mock the unpin page

  ON_CALL(*MockRequestFactory::_requestFactory, unpinPage).WillByDefault(testing::Invoke(
      [&](pdb::PDBLoggerPtr &myLogger, int port, const std::string &address, bool onErr, size_t bytesForRequest,
          const std::function<bool(pdb::Handle<pdb::SimpleRequestResult>)> &processResponse,
          PDBSetPtr &set, size_t pageNum, bool isDirty) {

        // do the bookkeeping
        {
          unique_lock<std::mutex> lock(m);

          // mark it as unpinned
          pinned[pageNum] = false;

          // expect it to be pinned when you return it
          EXPECT_TRUE(pinned[pageNum]);
        }

        const pdb::UseTemporaryAllocationBlock tempBlock{1024};

        // check the page
        EXPECT_GE(pageNum, 0);
        EXPECT_LE(pageNum, numPages);

        EXPECT_TRUE(set->getSetName() == "set1");
        EXPECT_TRUE(set->getDBName() == "DB");

        // make the page
        pdb::Handle<pdb::SimpleRequestResult> returnPageRequest = pdb::makeObject<pdb::SimpleRequestResult>(true, "");

        // return true since we assume this succeeded
        return processResponse(returnPageRequest);
      }
  ));

  // it should call send object exactly twice
  EXPECT_CALL(*MockRequestFactory::_requestFactory, unpinPage).Times(0);

  /// 3. Mock the freeze size

  ON_CALL(*MockRequestFactory::_requestFactory, freezeSize).WillByDefault(testing::Invoke(
      [&](pdb::PDBLoggerPtr &myLogger, int port, const std::string address, bool onErr,
          size_t bytesForRequest, const std::function<bool(pdb::Handle<pdb::BufFreezeRequestResult>)> &processResponse,
          pdb::PDBSetPtr setPtr, size_t pageNum, size_t numBytes) {

        // do the bookkeeping
        {
          unique_lock<std::mutex> lock(m);

          // expect not to be frozen
          EXPECT_FALSE(frozen[pageNum]);

          // mark it as frozen
          frozen[pageNum] = true;
        }

        const pdb::UseTemporaryAllocationBlock tempBlock{1024};

        // check the page
        EXPECT_GE(pageNum, 0);
        EXPECT_LE(pageNum, numPages);
        EXPECT_TRUE(setPtr->getSetName() == "set1");
        EXPECT_TRUE(setPtr->getDBName() == "DB");

        // make the page
        pdb::Handle<pdb::BufFreezeRequestResult> returnPageRequest = pdb::makeObject<pdb::BufFreezeRequestResult>(true);

        // return true since we assume this succeeded
        return processResponse(returnPageRequest);
      }
  ));

  // expected to be called 33 times
  EXPECT_CALL(*MockRequestFactory::_requestFactory, freezeSize).Times(numPages);

  /// 4. Mock the pin page

  ON_CALL(*MockRequestFactory::_requestFactory, pinPage).WillByDefault(testing::Invoke(
      [&](pdb::PDBLoggerPtr &myLogger, int port, const std::string &address, bool onErr,
          size_t bytesForRequest, const std::function<bool(pdb::Handle<pdb::BufPinPageResult>)> &processResponse,
          const pdb::PDBSetPtr &setPtr, size_t pageNum) {

        // do the bookkeeping
        {
          unique_lock<std::mutex> lock(m);

          // mark it as unpinned
          pinned[pageNum] = true;
        }

        const pdb::UseTemporaryAllocationBlock tempBlock{1024};

        // check the page
        EXPECT_GE(pageNum, 0);
        EXPECT_LE(pageNum, numPages);
        EXPECT_TRUE(setPtr->getSetName() == "set1");
        EXPECT_TRUE(setPtr->getDBName() == "DB");

        // make the page
        pdb::Handle<pdb::BufPinPageResult> returnPageRequest = pdb::makeObject<pdb::BufPinPageResult>(pageNum * maxPageSize, true);

        // return true since we assume this succeeded
        return processResponse(returnPageRequest);
      }
  ));

  // expected to be called twice
  EXPECT_CALL(*MockRequestFactory::_requestFactory, pinPage).Times(0);

  /// 5. Mock return page

  ON_CALL(*MockRequestFactory::_requestFactory, returnPage).WillByDefault(testing::Invoke(
      [&](pdb::PDBLoggerPtr &myLogger, int port, const std::string &address, bool onErr,
          size_t bytesForRequest, const std::function<bool(pdb::Handle<pdb::SimpleRequestResult>)> &processResponse,
          std::string setName, std::string dbName, size_t pageNum, bool isDirty) {

        // do the bookkeeping
        {
          unique_lock<std::mutex> lock(m);

          // expect it to be pinned when you return it
          EXPECT_TRUE(pinned[pageNum]);

          // mark it as unpinned
          pinned[pageNum] = false;
        }

        const pdb::UseTemporaryAllocationBlock tempBlock{1024};

        // check the page
        EXPECT_GE(pageNum, 0);
        EXPECT_LE(pageNum, numPages);
        EXPECT_TRUE(setName == "set1");
        EXPECT_TRUE(dbName == "DB");

        // make the page
        pdb::Handle<pdb::SimpleRequestResult> returnPageRequest = pdb::makeObject<pdb::SimpleRequestResult>(true, "");

        // return true since we assume this succeeded
        return processResponse(returnPageRequest);
      }
  ));

  // it should call send object exactly 98 times
  EXPECT_CALL(*MockRequestFactory::_requestFactory, returnPage).Times(testing::AtLeast(1));

  // do the testing
  {
    // the page sizes we are testing
    std::vector<size_t> pageSizes {8, 16, 32, 64, 128};

    // used to make sure we freeze only once
    std::vector<bool> firstTime(numPages, true);

    // lock
    std::mutex firstTimeMutex;

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

    // generate the pages
    PDBSetPtr set = make_shared<PDBSet>("DB", "set1");

    for(int t = 0; t < numThreads; ++t) {

      // grab a worker
      PDBWorkerPtr myWorker = workers->getWorker();

      // the thread
      int tmp = t;

      // start the thread
      PDBWorkPtr myWork = make_shared<pdb::GenericWork>([&, tmp](PDBBuzzerPtr callerBuzzer) {

        int myThread = tmp;
        int myThreadClamp = ((myThread + 1) * 100) % 127;

        // generate the page indices
        std::vector<uint64_t> pageIndices;
        for(int i = 0; i < numRequestsPerPage; ++i) {
          for(int j = 0; j < numPages; ++j) {
            pageIndices.emplace_back(j);
          }
        }

        // shuffle the page indices
        auto seed = std::chrono::system_clock::now().time_since_epoch().count();
        shuffle (pageIndices.begin(), pageIndices.end(), std::default_random_engine(seed));

        sync++;
        while (sync != numThreads) {}
        for(auto it : pageIndices) {

          // grab the page
          auto page = myMgr.getPage(set, it);

          firstTimeMutex.lock();

          if(firstTime[it]) {

            // freeze the size
            page->freezeSize(pageSizes[it % 5]);

            for(int k = 0; k < numThreads; ++k) {
              // set the first numThreads bytes to 0
              ((char *) page->getBytes())[k] = 0;
            }

            // mark as dirty
            page->setDirty();

            // mark it as false
            firstTime[it] = false;
          }

          firstTimeMutex.unlock();

          // increment the page
          ((char *) page->getBytes())[myThread] = (char) ((((char *) page->getBytes())[myThread] + 1) % myThreadClamp);

          // set as dirty
          page->setDirty();
        }

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

    for(uint64_t i = 0; i < numPages; ++i) {

      // the page
      auto page = myMgr.getPage(set, i);

      for(int t = 0; t < numThreads; ++t) {

        int myThreadClamp = ((t + 1) * 100) % 127;

        // check them
        EXPECT_EQ(((char*) page->getBytes())[t], (numRequestsPerPage % myThreadClamp));
      }
    }
  }

  // just to remove the mock object
  MockRequestFactory::_requestFactory = nullptr;
  myMgr.parent = nullptr;
}

}