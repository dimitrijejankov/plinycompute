#include <utility>
#include <Join8SideReader.h>
#include <PDBVector.h>

using namespace pdb::matrix_3d;

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
  for(int i = 0; i < 100; ++i) {
    toSend[i].reserve(100);
  }
}

void pdb::Join8SideReader::run() {

  // get plan from the page
  auto* record = (Record<JoinPlannerResult>*) planPage->getBytes();
  planResult = record->getRootObject();

  PDBPageHandle page;
  while((page = pageSet->getNextPage(workerID)) != nullptr) {

    // get the bytes of the page
    page->repin();

    //
    auto *recordCopy = (Record<Vector<Handle<MatrixBlock3D>>> *) page->getBytes();
    auto records = recordCopy->getRootObject();

    for(int i = 0; i < records->size(); ++i) {

      // get the matrix
      auto matrix = (*records)[i];

      // get the tid
      auto tid = (*planResult->records)[*matrix->getKey()];

      // get the node based on the tid
      auto node = (*planResult->mapping)[tid];

      toSend[node].push_back(matrix);
    }

    // send them all
    for(int i = 0; i < numNodes; ++i) {
      auto id = (*joinSideSenders)[i]->queueToSend(&toSend[i]);
      (*joinSideSenders)[i]->waitToFinish(id);
    }
  }
}
