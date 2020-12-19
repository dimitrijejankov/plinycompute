#include <JoinPlanner.h>
#include <GeneticJoinPlanner.h>
#include <UseTemporaryAllocationBlock.h>
#include <PDBVector.h>
#include <JoinPlannerResult.h>
#include <GreedyPlanner.h>
#include "../../../../applications/TestConvolution/sharedLibraries/headers/MatrixBlockMeta3D.h"

using namespace pdb::matrix_3d;

pdb::JoinPlanner::JoinPlanner(uint32_t numNodes,
                              uint32_t numThreads,
                              std::unordered_map<Join3KeyPipeline::key, pair<int32_t, int32_t>, Join3KeyPipeline::HashFunction> &nodeRecords0,
                              std::unordered_map<Join3KeyPipeline::key, pair<int32_t, int32_t>, Join3KeyPipeline::HashFunction> &nodeRecords1,
                              std::unordered_map<Join3KeyPipeline::key, pair<int32_t, int32_t>, Join3KeyPipeline::HashFunction> &nodeRecords2,
                              std::vector<Join3KeyPipeline::joined_record> &joined,
                              std::vector<std::vector<int32_t>> &aggGroups) : numNodes(numNodes),
                                                                                      numThreads(numThreads),
                                                                                      nodeRecords0(nodeRecords0),
                                                                                      nodeRecords1(nodeRecords1),
                                                                                      nodeRecords2(nodeRecords2),
                                                                                      joined(joined),
                                                                                      aggGroups(aggGroups) {
}

void pdb::JoinPlanner::doPlanning(const PDBPageHandle &page) {

  // we need this for the planner
  std::vector<char> side_tids;

  // allocate the side tids per node
  side_tids.resize(numNodes * (nodeRecords0.size() + nodeRecords1.size() + nodeRecords2.size()) );

  // mark it as true
  for(auto &r : nodeRecords0) {
    side_tids[r.second.second * numNodes + r.second.first] = true;
  }

  for(auto &r : nodeRecords1) {
    side_tids[r.second.second * numNodes + r.second.first] = true;
  }

  for(auto &r : nodeRecords2) {
    side_tids[r.second.second * numNodes + r.second.first] = true;
  }

  pdb::GreedyPlanner::costs_t c{};
  c.agg_cost = 1;
  c.join_cost = 1;
  c.join_rec_size = 1;
  c.send_coef = 1;
  c.rhs_input_rec_size = 1;
  c.lhs_input_rec_size = 1;
  c.aggregation_rec_size = 1;

    // init the planner run the agg only planner
  pdb::GreedyPlanner planner(numNodes, c, side_tids, aggGroups, joined);

  // run for a number of iterations
  planner.run_agg_first_only();

  // print it
  planner.print();

  // get the result of the planning
  auto result = planner.get_result();

  // write stuff to the page
  UseTemporaryAllocationBlock blk{page->getBytes(), page->getSize()};

  pdb::Handle<JoinPlannerResult> out = pdb::makeObject<JoinPlannerResult>();

//  // this is the stuff we need to execute the query
//  out->mapping = pdb::makeObject<pdb::Vector<int32_t>>(joined.size(), joined.size());
//  out->recordToNode = pdb::makeObject<pdb::Vector<bool>>(nodeRecords.size() * numNodes, nodeRecords.size() * numNodes);
//  out->joinedRecords = pdb::makeObject<pdb::Vector<Join3KeyPipeline::joined_record>>(joined.size(), joined.size());
//  out->records = pdb::makeObject<pdb::Map<MatrixBlockMeta3D, int32_t>>();
//
//  // zero out the record to node
//  bzero(out->recordToNode->c_ptr(), sizeof(bool) * nodeRecords.size() * numNodes);
//
//  // go through the result
//  for(int jg = 0; jg < result.size(); ++jg) {
//
//    auto node = result[jg];
//
//    // set the mapping
//    (*out->mapping)[jg] = node;
//
//    // get the joined record
//    auto &record = joined[jg];
//
//    // copy the stuff
//    (*out->joinedRecords)[jg].first = record.first;
//    (*out->joinedRecords)[jg].second = record.second;
//    (*out->joinedRecords)[jg].third = record.third;
//    (*out->joinedRecords)[jg].fourth = record.fourth;
//    (*out->joinedRecords)[jg].fifth = record.fifth;
//    (*out->joinedRecords)[jg].sixth = record.sixth;
//    (*out->joinedRecords)[jg].seventh = record.seventh;
//    (*out->joinedRecords)[jg].eigth = record.eigth;
//
//    // do the record to node mapping
//    (*out->recordToNode)[record.first * numNodes + node] = true;
//    (*out->recordToNode)[record.second * numNodes + node] = true;
//    (*out->recordToNode)[record.third * numNodes + node] = true;
//    (*out->recordToNode)[record.fourth * numNodes + node] = true;
//    (*out->recordToNode)[record.fifth * numNodes + node] = true;
//    (*out->recordToNode)[record.sixth * numNodes + node] = true;
//    (*out->recordToNode)[record.seventh * numNodes + node] = true;
//    (*out->recordToNode)[record.eigth * numNodes + node] = true;
//  }
//
//  // copy the records with tid mappings
//  for(auto &r : nodeRecords) {
//    (*out->records)[MatrixBlockMeta3D(r.first.first, r.first.second, r.first.third)] = r.second.first;
//  }

  // set the root object
  getRecord(out);
}