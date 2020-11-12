#include <TRABlock.h>
#include <mkl.h>
#include "MM3DMultiplyStage.h"
#include "MM3DState.h"
#include "ExJob.h"
#include "GenericWork.h"

pdb::MM3DMultiplyStage::MM3DMultiplyStage(int32_t n, int32_t num_nodes, int32_t num_threads) :
    PDBPhysicalAlgorithmStage(*(_sink),
                              *(_sources),
                              *(_finalTupleSet),
                              *(_secondarySources),
                              *(_setsToMaterialize)), idx{.num_nodes = num_nodes,
                                                          .num_threads = num_threads,
                                                          .n = n} {}

bool pdb::MM3DMultiplyStage::setup(const pdb::Handle<pdb::ExJob> &job,
                                   const pdb::PDBPhysicalAlgorithmStatePtr &state,
                                   const shared_ptr<pdb::PDBStorageManagerBackend> &storage,
                                   const string &error) {
  // cast the state
  auto s = dynamic_pointer_cast<pdb::MM3DState>(state);

  s->outQueues = std::make_shared<std::vector<PDBPageQueuePtr>>();

  return true;
}

bool pdb::MM3DMultiplyStage::run(const pdb::Handle<pdb::ExJob> &job,
                                 const pdb::PDBPhysicalAlgorithmStatePtr &state,
                                 const shared_ptr<pdb::PDBStorageManagerBackend> &storage,
                                 const string &error) {

  // cast the state
  auto s = dynamic_pointer_cast<pdb::MM3DState>(state);

  // success indicator
  atomic_bool success;

  // lhs and rhs pages
  vector<PDBPageHandle> lhsPages;
  vector<PDBPageHandle> rhsPages;

  PDBPageHandle page;
  while((page = s->feedingPageSetLHS->getNextPage(0)) != nullptr) {

    cout << "Fetched page....\n";

    page->repin();
    lhsPages.push_back(page);

    // get the vector from the page
    auto &vec = *(((Record<Vector<Handle<TRABlock>>> *) page->getBytes())->getRootObject());

    // generate the index
    for(int i = 0; i < vec.size(); ++i) {

      auto &meta = *vec[i]->metaData;
      s->indexLHS[{meta.getIdx0(), meta.getIdx1()}] = &(*vec[i]);
      vec[i]->print_meta();
    }
  }

  while((page = s->feedingPageSetRHS->getNextPage(0)) != nullptr) {

    cout << "Fetched page....\n";
    page->repin();
    rhsPages.push_back(page);

    // get the vector from the page
    auto &vec = *(((Record<Vector<Handle<TRABlock>>> *) page->getBytes())->getRootObject());

    // generate the index
    for(int i = 0; i < vec.size(); ++i) {

      auto &meta = *vec[i]->metaData;
      s->indexRHS[{meta.getIdx0(), meta.getIdx1()}] = &(*vec[i]);
      vec[i]->print_meta();
    }
  }

  // create the buzzer
  atomic_int multiplyDone;
  multiplyDone = 0;
  PDBBuzzerPtr multiplyBuzzer = make_shared<PDBBuzzer>([&](PDBAlarm myAlarm, atomic_int &cnt) {

    // did we fail?
    if (myAlarm == PDBAlarm::GenericError) {
      success = false;
    }

    // we are done here
    cnt++;
  });

  auto bufferManager = storage->getFunctionalityPtr<PDBBufferManagerInterface>();
  auto nodeID = job->thisNode;
  for(int thread = 0; thread < job->numberOfProcessingThreads; ++thread) {

    // make the work
    PDBWorkPtr myWork = std::make_shared<pdb::GenericWork>([&, thread](const PDBBuzzerPtr& callerBuzzer) {

      auto [x, y, z] = idx.get_coords(nodeID, thread);
      cout << "X : " << x << " Y : " << y << " Z : " << z << "\n";

      // get a new page
      auto currentPage = bufferManager->getPage();
      makeObjectAllocatorBlock(currentPage->getBytes(), currentPage->getSize(), true);

      auto lhs = s->indexLHS.find({x, z});
      auto rhs = s->indexRHS.find({z, y});

      if(lhs == s->indexLHS.end() || rhs == s->indexRHS.end()) {
        std::cout << "Bad shit!!!\n";
      }
      else {
        std::cout << "Found them\n";
      }

      try {

        auto &in1 = lhs->second->data;
        auto &in2 = rhs->second->data;
        auto &inMeta1 = lhs->second->metaData;
        auto &inMeta2 = rhs->second->metaData;

        // get the sizes
        uint32_t I = in1->dim0;
        uint32_t J = in2->dim1;
        uint32_t K = in1->dim1;

        // make an output
        Handle<TRABlock> out = makeObject<TRABlock>(inMeta1->getIdx0(), inMeta2->getIdx1(), 0, I, J, 1);

        // get the ptrs
        float *outData = out->data->data->c_ptr();
        float *in1Data = in1->data->c_ptr();
        float *in2Data = in2->data->c_ptr();

        // do the multiply
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, I, J, K, 1.0f, in1Data, K, in2Data, J, 0.0f, outData, J);

        // make this the root object
        getRecord(out);

      } catch (pdb::NotEnoughSpace &n) {
        throw runtime_error("Ok this is bad!");
      }

      // invalid
      makeObjectAllocatorBlock(1024, true);

      // signal that the run was successful
      callerBuzzer->buzz(PDBAlarm::WorkAllDone, multiplyDone);
    });

    // run the work
    storage->getWorker()->execute(myWork, multiplyBuzzer);
  }

  while(multiplyDone < job->numberOfProcessingThreads) {
    multiplyBuzzer->wait();
  }

  return true;
}

void pdb::MM3DMultiplyStage::cleanup(const pdb::PDBPhysicalAlgorithmStatePtr &state,
                                     const shared_ptr<pdb::PDBStorageManagerBackend> &storage) {
}

const pdb::PDBSinkPageSetSpec *pdb::MM3DMultiplyStage::_sink = nullptr;
const pdb::Vector<pdb::PDBSourceSpec> *pdb::MM3DMultiplyStage::_sources = nullptr;
const pdb::String *pdb::MM3DMultiplyStage::_finalTupleSet = nullptr;
const pdb::Vector<pdb::Handle<pdb::PDBSourcePageSetSpec>> *pdb::MM3DMultiplyStage::_secondarySources = nullptr;
const pdb::Vector<pdb::PDBSetObject> *pdb::MM3DMultiplyStage::_setsToMaterialize = nullptr;