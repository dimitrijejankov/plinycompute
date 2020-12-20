#include <utility>
#include <Join8SideReader.h>
#include <PDBVector.h>

using namespace pdb::matrix;

pdb::Join8SideReader::Join8SideReader(pdb::PDBAbstractPageSetPtr pageSet,
                                      int32_t workerID,
                                      int32_t numNodes,
                                      std::shared_ptr<std::vector<Join8SideSenderPtr>> joinSideSenders,
                                      PDBPageHandle page) : pageSet(std::move(pageSet)),
                                                            workerID(workerID),
                                                            numNodes(numNodes),
                                                            joinSideSenders(std::move(joinSideSenders)),
                                                            planPage(std::move(page)) {

  // resize so we have stuff to send
  toSend.resize(numNodes);
  for(int i = 0; i < numNodes; ++i) {
    toSend[i].reserve(100);
  }
}

void pdb::Join8SideReader::run() {

  // get plan from the page
  auto* record = (Record<JoinPlannerResult>*) planPage->getBytes();
  planResult = record->getRootObject();

  PDBPageHandle page;
  while((page = pageSet->getNextPage(workerID)) != nullptr) {

    // clear all vectors
    for(auto &s : toSend) {
      s.clear();
    }

    // get the bytes of the page
    page->repin();

    //
    auto *recordCopy = (Record<Vector<Handle<MatrixBlock>>> *) page->getBytes();
    auto records = recordCopy->getRootObject();

    for(int i = 0; i < records->size(); ++i) {

      // get the matrix
      auto matrix = (*records)[i];

      // get the tid
      auto tid = (*planResult->records0)[*matrix->getKey()];

      // get the node based on the tid
      auto node = &(*planResult->record_mapping)[tid * numNodes];

      // go and figure out what nodes we need to put his on
      for(int n = 0; n < numNodes; ++n) {

        // if we need to send this to this node
        if(node[n]){
          toSend[n].push_back(matrix);
        }
      }
    }

    // send them all
    for(int i = 0; i < numNodes; ++i) {
      auto id = (*joinSideSenders)[i]->queueToSend(&toSend[i]);
      (*joinSideSenders)[i]->waitToFinish(id);
    }
  }
}
