#include <JoinPlanner.h>
#include <GeneticJoinPlanner.h>
#include <UseTemporaryAllocationBlock.h>
#include <PDBVector.h>
#include <JoinPlannerResult.h>
#include "../../../../applications/TestConvolution/sharedLibraries/headers/MatrixBlockMeta3D.h"

using namespace pdb::matrix_3d;

pdb::JoinPlanner::JoinPlanner(uint32_t numNodes,
                              uint32_t numThreads,
                              std::unordered_map<EightWayJoinPipeline::key, pair<int32_t, int32_t>, EightWayJoinPipeline::HashFunction> &nodeRecords,
                              std::vector<EightWayJoinPipeline::joined_record> &joined) : numNodes(numNodes),
                                                                                          numThreads(numThreads),
                                                                                          nodeRecords(nodeRecords),
                                                                                          joined(joined) {
}

void pdb::JoinPlanner::doPlanning(const PDBPageHandle &page) {

  // we need this for the planner
  std::vector<std::vector<bool>> side_tids;

  // allocate the side tids per node
  side_tids.resize(numNodes);
  for(int i = 0; i < numNodes; ++i) {
    side_tids[i].resize(nodeRecords.size());
  }

  // mark it as true
  for(auto &r : nodeRecords) {
    side_tids[r.second.second][r.second.first] = true;
  }

  pdb::GeneticJoinPlanner planner(nodeRecords.size(),
                                  joined.size(),
                                  numNodes,
                                  1, // need to get this parameter
                                  side_tids,
                                  joined,
                                  40);

  // run for a number of iterations
  planner.run(100);

  // get the result of the planning
  auto result = planner.getResult();

  // write stuff to the page
  UseTemporaryAllocationBlock blk{page->getBytes(), page->getSize()};

  pdb::Handle<JoinPlannerResult> out = pdb::makeObject<JoinPlannerResult>();

  // this is the stuff we need to execute the query
  out->mapping = pdb::makeObject<pdb::Vector<int32_t>>(joined.size(), joined.size());
  out->recordToNode = pdb::makeObject<pdb::Vector<bool>>(nodeRecords.size() * numNodes, nodeRecords.size() * numNodes);
  out->joinedRecords = pdb::makeObject<pdb::Vector<EightWayJoinPipeline::joined_record>>(joined.size(), joined.size());
  out->records = pdb::makeObject<pdb::Map<MatrixBlockMeta3D, int32_t>>();

  // zero out the record to node
  bzero(out->recordToNode->c_ptr(), sizeof(bool) * nodeRecords.size() * numNodes);

  // go through the result
  for(int jg = 0; jg < result.size(); ++jg) {

    auto node = result[jg];

    // set the mapping
    (*out->mapping)[jg] = node;

    // get the joined record
    auto record = joined[jg];

    // copy the stuff
    std::get<0>((*out->joinedRecords)[jg]) = std::get<0>(record);
    std::get<1>((*out->joinedRecords)[jg]) = std::get<1>(record);
    std::get<2>((*out->joinedRecords)[jg]) = std::get<2>(record);
    std::get<3>((*out->joinedRecords)[jg]) = std::get<3>(record);
    std::get<4>((*out->joinedRecords)[jg]) = std::get<4>(record);
    std::get<5>((*out->joinedRecords)[jg]) = std::get<5>(record);
    std::get<6>((*out->joinedRecords)[jg]) = std::get<6>(record);
    std::get<7>((*out->joinedRecords)[jg]) = std::get<7>(record);

    // do the record to node mapping
    (*out->recordToNode)[std::get<0>(record) * numNodes + node] = true;
    (*out->recordToNode)[std::get<1>(record) * numNodes + node] = true;
    (*out->recordToNode)[std::get<2>(record) * numNodes + node] = true;
    (*out->recordToNode)[std::get<3>(record) * numNodes + node] = true;
    (*out->recordToNode)[std::get<4>(record) * numNodes + node] = true;
    (*out->recordToNode)[std::get<5>(record) * numNodes + node] = true;
    (*out->recordToNode)[std::get<6>(record) * numNodes + node] = true;
    (*out->recordToNode)[std::get<7>(record) * numNodes + node] = true;
  }

  // copy the records with tid mappings
  for(auto &r : nodeRecords) {
    (*out->records)[MatrixBlockMeta3D(r.first)] = r.second.first;
  }

  // set the root object
  getRecord(out);
}