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
  auto myMgr = storage->getFunctionalityPtr<PDBBufferManagerInterface>();

  s->outQueues = std::make_shared<std::vector<PDBPageQueuePtr>>();
  for(int i = 0; i < job->numberOfProcessingThreads; ++i) { s->outQueues->emplace_back(std::make_shared<PDBPageQueue>()); }

  /// 1. Make outgoing connections to other nodes

  // make the object
  UseTemporaryAllocationBlock tmp{1024};
  pdb::Handle<SerConnectToRequest> connectionID = pdb::makeObject<SerConnectToRequest>(job->computationID,
                                                                                       job->jobID,
                                                                                       job->thisNode,
                                                                                       0);

  // init the vector for the left sides
  s->aggOut = std::make_shared<std::vector<PDBCommunicatorPtr>>();
  for (int thread = 0; thread < job->numberOfProcessingThreads; thread++) {

    auto [x, y, z] = idx.get_coords(job->thisNode, thread);
    std::cout << "Outgoing x : " << x << " y : " << y << " z : " << z << '\n';

    // skip the one that are on the edge of the plane
    if(z == 0) {
      s->aggOut->push_back(nullptr);
      continue;
    }

    // get the node id
    auto [node, _] = idx.get(x, y, z - 1);
    std::cout << "... to node : " << node << '\n';

    // skip this node to avoid self connects
    if(node == job->thisNode) {
      s->aggOut->push_back(nullptr);
      continue;
    }

    // set the task id
    connectionID->taskID = idx.getGlobal(x, y, z - 1);

    // connect to the node
    s->aggOut->push_back(myMgr->connectTo(job->nodes[node]->address,
                                          job->nodes[node]->backendPort,
                                          connectionID));
  }

  /// 2. Get the incoming connections to this node.

  // wait for left side connections
  connectionID->taskID = AGG_TASK;
  s->aggIn = std::make_shared<std::vector<PDBCommunicatorPtr>>();
  for (int thread = 0; thread < job->numberOfProcessingThreads; thread++) {

    auto [x, y, z] = idx.get_coords(job->thisNode, thread);
    std::cout << "Incoming x : " << x << " y : " << y << " z : " << z << '\n';

    // skip the one that are on the edge of the plane
    if(z == idx.get_side() - 1) {
      s->aggIn->push_back(nullptr);
      continue;
    }

    // get the node id
    auto [node, _] = idx.get(x, y, z + 1);
    std::cout << "... to node : " << node << '\n';

    // set the node id
    connectionID->nodeID = node;

    // skip this node to avoid self connects
    if(node == job->thisNode) {
      s->aggOut->push_back(nullptr);
      continue;
    }

    // set the task id
    connectionID->taskID = idx.getGlobal(x, y, z);

    // wait for the connection
    s->aggIn->push_back(myMgr->waitForConnection(connectionID));

    // check if the socket is open
    if (s->aggIn->back()->isSocketClosed()) {

      // log the error
      s->logger->error("Socket for the left side is closed");

      return false;
    }
  }

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

  // prep the left pages
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

  // prep the right pages
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

  // do the multiply
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
        Handle<Vector<Handle<TRABlock>>> vec = makeObject<Vector<Handle<TRABlock>>>();
        Handle<TRABlock> out = makeObject<TRABlock>(inMeta1->getIdx0(), inMeta2->getIdx1(), 0, I, J, 1);

        // get the ptrs
        float *outData = out->data->data->c_ptr();
        float *in1Data = in1->data->c_ptr();
        float *in2Data = in2->data->c_ptr();

        // do the multiply
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, I, J, K, 1.0f, in1Data, K, in2Data, J, 0.0f, outData, J);

        // make this the root object
        vec->push_back(out);
        getRecord(vec);

      } catch (pdb::NotEnoughSpace &n) {
        throw runtime_error("Ok this is bad!");
      }

      // invalid
      makeObjectAllocatorBlock(1024, true);

      // move it to the right queue
      (*s->outQueues)[thread]->enqueue(currentPage);

      // signal that the run was successful
      callerBuzzer->buzz(PDBAlarm::WorkAllDone, multiplyDone);
    });

    // run the work
    storage->getWorker()->execute(myWork, multiplyBuzzer);
  }

  // create the buzzer
  atomic_int addDone;
  addDone = 0;
  PDBBuzzerPtr addBuzzer = make_shared<PDBBuzzer>([&](PDBAlarm myAlarm, atomic_int &cnt) {

    // did we fail?
    if (myAlarm == PDBAlarm::GenericError) {
      success = false;
    }

    // we are done here
    cnt++;
  });

  // run a bunch of threads to merge stuff
  auto rps = std::make_shared<PDBRandomAccessPageSet>(bufferManager);
  for(int thread = 0; thread < job->numberOfProcessingThreads; ++thread) {

    // make the work
    PDBWorkPtr myWork = std::make_shared<pdb::GenericWork>([&, thread](const PDBBuzzerPtr& callerBuzzer) {

      // get my coordinates
      auto [x, y, z] = idx.get_coords(nodeID, thread);

      // depending on z we either forward one we have
      PDBPageHandle p1;
      if(z != idx.get_side() - 1) {

        // get the first page
        (*s->outQueues)[thread]->wait_dequeue(p1);

        // get the second page
        PDBPageHandle p2;
        (*s->outQueues)[thread]->wait_dequeue(p2);

        // get the vector from the page
        auto &pp1 = *((*(((Record<Vector<Handle<TRABlock>>> *) p1->getBytes())->getRootObject()))[0]);
        auto &pp2 = *((*(((Record<Vector<Handle<TRABlock>>> *) p2->getBytes())->getRootObject()))[0]);

        // sum up all the
        cout << "Added " << pp1.getDim0() << " " <<  pp2.getDim1() << '\n';
        for(int i = 0; i < pp1.getDim0() * pp2.getDim1(); ++i) {
          pp1.data->data->c_ptr()[i] += pp2.data->data->c_ptr()[i];
        }

        // unpin the page
        p2->unpin();
      }
      else {

        // get the first page
        (*s->outQueues)[thread]->wait_dequeue(p1);
      }

      // check if we are keeping this page
      if(z != 0) {

        cout << "sent!\n";
        // the node and thread where we need to send this
        auto [dstNode, dstThread] = idx.get(x, y, z - 1);

        // are we on the same node if it is just forward to the right queue
        if(dstNode == job->thisNode) {

          // forward the page to a local thread
          (*s->outQueues)[dstThread]->enqueue(p1);
        }
        else {

          // get the right communicator
          auto &comm = (*s->aggOut)[thread];

          // set the record
          string error;
          auto pp1 = ((Record<Vector<Handle<TRABlock>>> *) p1->getBytes());
          if(!comm->sendBytes(pp1, pp1->numBytes(), error)) {
            cout << "Failed to send!";
          }
        }
      }
      else {

        cout << "kept!\n";
        // we keep this page promote it to a set page
        rps->pushPage(p1);
      }

      // signal that the run was successful
      callerBuzzer->buzz(PDBAlarm::WorkAllDone, addDone);
    });

    // run the work
    storage->getWorker()->execute(myWork, addBuzzer);
  }

  // create the buzzer
  atomic_int recvDone;
  recvDone = 0;
  PDBBuzzerPtr recvBuzzer = make_shared<PDBBuzzer>([&](PDBAlarm myAlarm, atomic_int &cnt) {

    // did we fail?
    if (myAlarm == PDBAlarm::GenericError) {
      success = false;
    }

    // we are done here
    cnt++;
  });

  // do the receiving
  for(int thread = 0; thread < job->numberOfProcessingThreads; ++thread) {

    // make the work
    PDBWorkPtr myWork = std::make_shared<pdb::GenericWork>([&, thread](const PDBBuzzerPtr& callerBuzzer) {

      // get the communicator and check if need to recv something
      auto comm = (*s->aggIn)[thread];
      if(comm == nullptr){
        callerBuzzer->buzz(PDBAlarm::WorkAllDone, recvDone);
        return;
      }

      // get the page
      auto recvPage = bufferManager->getPage();

      // try to receive
      string error;
      if(!comm->receiveBytes(recvPage->getBytes(), error)) {
        cout << "We failed to receive.";
      }

      cout << "recieved \n";

      // move the page into the queue
      (*s->outQueues)[thread]->enqueue(recvPage);

      // signal that the run was successful
      callerBuzzer->buzz(PDBAlarm::WorkAllDone, recvDone);
    });

    // run the work
    storage->getWorker()->execute(myWork, recvBuzzer);
  }

  while(multiplyDone < job->numberOfProcessingThreads) {
    multiplyBuzzer->wait();
  }

  while(addDone < job->numberOfProcessingThreads) {
    addBuzzer->wait();
  }

  while(recvDone < job->numberOfProcessingThreads) {
    recvBuzzer->wait();
  }

  storage->materializePageSet(rps, {"myData", "C"});

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