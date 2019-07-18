#include <utility>

//
// Created by dimitrije on 4/5/19.
//

#include <PDBPageSelfReceiver.h>

#include "PDBPageSelfReceiver.h"

pdb::PDBPageSelfReceiver::PDBPageSelfReceiver(pdb::PDBPageQueuePtr queue,
                                              pdb::PDBFeedingPageSetPtr pageSet,
                                              pdb::PDBBufferManagerInterfacePtr bufferManager) : queue(std::move(queue)),
                                                                                                 pageSet(std::move(pageSet)),
                                                                                                 bufferManager(std::move(bufferManager)) {}

bool pdb::PDBPageSelfReceiver::run() {

  PDBPageHandle page;
  do {

    // kill the page if there is any
    page = nullptr;

    // get a page
    queue->wait_dequeue(page);

    // if we got a page from the queue
    if(page != nullptr) {

      // repin the page
      page->repin();

      // the size of the page
      auto pageSize = ((Record<Object> *) page->getBytes())->numBytes();

      // the output page
      auto outPage = bufferManager->getPage(pageSize);

      // copy the memory of the input page to the input page // TODO this copy might not be optimal maybe a page swap or a smarter freeze would be better.
      memcpy(outPage->getBytes(), page->getBytes(), pageSize);

      // unpin them both
      outPage->unpin();

      // feed the page into the page set...
      pageSet->feedPage(outPage);
    }

  } while (page != nullptr);

  // kill the page
  page = nullptr;

  // finish feeding the page set
  pageSet->finishFeeding();

  // we are done here everything worked
  return true;
}
