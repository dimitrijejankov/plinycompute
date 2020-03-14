#include <EightWayJoinPipeline.h>
#include <PDBPageHandle.h>
#include <PDBAbstractPageSet.h>
#include <PDBVector.h>
#include "../../../../../applications/TestConvolution/sharedLibraries/headers/MatrixBlockMeta3D.h"

void pdb::EightWayJoinPipeline::runSide(int node) {

  //
  auto pageSet = keySourcePageSets[node];

  // get all the pages from the page set
  PDBPageHandle page;
  while ((page = pageSet->getNextPage(0)) != nullptr) {

    // repin the page
    page->repin();

    // get the record
    auto curRec = (Record<Vector<Handle<matrix_3d::MatrixBlockMeta3D>>> *) page->getBytes();

    // get the root vector
    auto inVec = curRec->getRootObject();

    // go through the vector
    for(int i = 0; i < inVec->size(); ++i) {
      std::cout << (*inVec)[i]->x_id << " " << (*inVec)[i]->y_id << " " << (*inVec)[i]->z_id << '\n';
    }
  }

}
