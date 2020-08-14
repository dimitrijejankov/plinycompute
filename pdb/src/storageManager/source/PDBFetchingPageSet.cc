#include <PDBCommunicator.h>
#include <PDBFetchingPageSet.h>

#include <utility>
#include <PDBWorker.h>
#include <GenericWork.h>
#include <StoFetchNextPage.h>
#include <HeapRequest.h>
#include <StoFetchNextPageResult.h>

namespace pdb {

PDBFetchingPageSet::~PDBFetchingPageSet() {

  // wait for the thread to finish
  tempBuzzer->wait();
}

pdb::PDBFetchingPageSet::PDBFetchingPageSet(PDBCommunicatorPtr comm,
                                            PDBStorageManagerPtr sto,
                                            PDBBufferManagerInterfacePtr buff,
                                            uint64_t numPages) : communicator(std::move(comm)),
                                                                 bufferManager(std::move(buff)),
                                                                 storageManager(std::move(sto)),
                                                                 numPages(numPages){

  // make the temp buzzer
  tempBuzzer = make_shared<PDBBuzzer>([&](PDBAlarm myAlarm) {

    // did we fail?
    if(myAlarm == PDBAlarm::GenericError) {
      success = false;
    }
  });

  // get a worker
  worker = this->bufferManager->getWorker();

  // init the page queue
  queue = std::make_shared<PDBPageQueue>();

  // start the thread
  PDBWorkPtr myWork = std::make_shared<pdb::GenericWork>([&](const PDBBuzzerPtr& callerBuzzer) {

    // create an allocation block to hold the response
    const UseTemporaryAllocationBlock tempBlock{1024};

    // get the logger for the buffer manager
    auto logger = this->bufferManager->getLogger();

    for(;;) {

      // info about the page
      uint64_t pageSize = 0;
      bool hasNext = false;

      // wait for the page
      success = RequestFactory::waitHeapRequest<StoFetchNextPageResult, bool>(logger, communicator, false,
        [&](const Handle<StoFetchNextPageResult>& result) {

          // check the result
          if (result == nullptr) {

            // log the error
            error = "Error could not get the next page";

            // return false
            return false;
          }

          // set the page parameters
          hasNext = result->hasNext;
          pageSize = result->pageSize;

          return true;
        });

      // if we failed
      if(!success) {
        break;
      }

      // if we don't have any page
      if(!hasNext) {

        // mark as true
        success = true;
        break;
      }

      // the page we want to store the stuff onto
      auto page = this->bufferManager->getPage(pageSize);

      // read the size
      auto readSize = RequestFactory::waitForBytes(logger,
                                                   this->communicator,
                                                   (char*) page->getBytes(),
                                                   page->getSize(),
                                                   error);

      std::cout << "Got page " << readSize << " \n";

      // unpin the page
      page->unpin();

      // put the page into the queue
      queue->enqueue(page);

      // did we read anything, if so break
      if (readSize == -1) {
        error = "Failed to get the the bytes";
        break;
      }
    }

    std::cout << "Finished \n";

    // buzz
    callerBuzzer->buzz(success ? PDBAlarm::WorkAllDone : PDBAlarm::GenericError);
  });

  // set the work
  worker->execute(myWork, tempBuzzer);
}

PDBPageHandle PDBFetchingPageSet::getNextPage(size_t) {

  // this was designed to be used for just one worker so we ignore the worker id
  PDBPageHandle pageHandle;
  if(curPage < numPages) {

    // dequeue the page
    queue->wait_dequeue(pageHandle);

    // increment the current page
    curPage++;

    // return it
    return pageHandle;
  }

  // return null
  return nullptr;
}

PDBPageHandle PDBFetchingPageSet::getNewPage() {
  throw runtime_error("Can not get a new page from the PDBFetchingPageSet.");
}

size_t PDBFetchingPageSet::getNumPages() {
  return numPages;
}

void PDBFetchingPageSet::resetPageSet() {
  curPage = 0;
}

void PDBFetchingPageSet::removePage(PDBPageHandle pageHandle) {
  throw runtime_error("Can not get remove a page from the PDBFetchingPageSet.");
}

size_t PDBFetchingPageSet::getMaxPageSize() {
  throw runtime_error("Can not set the maximum page size for PDBFetchingPageSet.");
}

}
