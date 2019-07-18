//
// Created by dimitrije on 4/5/19.
//

#ifndef PDB_PDBPAGESELFRECIEVER_H
#define PDB_PDBPAGESELFRECIEVER_H

#include <PDBFeedingPageSet.h>
#include "PageProcessor.h"

namespace pdb {

class PDBPageSelfReceiver;
using PDBPageSelfReceiverPtr = std::shared_ptr<PDBPageSelfReceiver>;

/**
 * This class is used to feed the pages that are created on this node to a particular @see PDBFeedingPageSet.
 * The pages are grabbed from the provided queue and we stop feeding the page set once we get a nullptr from the queue
 */
class PDBPageSelfReceiver {
public:

  PDBPageSelfReceiver(pdb::PDBPageQueuePtr queue,
                      pdb::PDBFeedingPageSetPtr pageSet,
                      pdb::PDBBufferManagerInterfacePtr bufferManager);

  /**
   * Basically starts the feeding of the page set. If something goes wrong returns false.
   * @return true if it succeeds false otherwise
   */
  bool run();

private:

  /**
   * The queue where we are getting the pages from
   */
  PDBPageQueuePtr queue;

  /**
   * The page set we are feeding with pages
   */
  pdb::PDBFeedingPageSetPtr pageSet;

  /**
   * The buffer manager
   */
  pdb::PDBBufferManagerInterfacePtr bufferManager;
};

}
#endif //PDB_PDBPAGESELFRECIEVER_H
