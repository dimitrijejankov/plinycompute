#include <utility>

#include <utility>

//
// Created by dimitrije on 4/5/19.
//

#include <PDBPageSelfReceiver.h>

#include "PDBPageSelfReceiver.h"

pdb::PDBPageSelfReceiver::PDBPageSelfReceiver(pdb::PDBPageQueuePtr queue,
                                              pdb::PDBFeedingPageSetPtr pageSet) : queue(std::move(queue)), pageSet(std::move(pageSet)) {}

bool pdb::PDBPageSelfReceiver::run() {

  PDBPageHandle page;
  do {

    // get a page
    queue->wait_dequeue(page);

    // if we got a page from the queue
    if(page != nullptr) {

      // feed the page into the page set...
      pageSet->feedPage(page);
    }

  } while (page != nullptr);

  // finish feeding the page set
  pageSet->finishFeeding();

  // we are done here everything worked
  return true;
}
