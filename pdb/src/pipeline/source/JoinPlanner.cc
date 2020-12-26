#include <JoinPlanner.h>
#include <UseTemporaryAllocationBlock.h>
#include <PDBVector.h>
#include <JoinPlannerResult.h>
#include <GreedyPlanner3.h>
#include <TRABlockMeta.h>
#include <cassert>

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

  pdb::GreedyPlanner3::costs_t c{};
  c.agg_cost = 1;
  c.join_cost = 1;
  c.join_rec_size = 1;
  c.send_coef = 1;
  c.rhs_input_rec_size = 1;
  c.lhs_input_rec_size = 1;
  c.aggregation_rec_size = 1;

    // init the planner run the agg only planner
  pdb::GreedyPlanner3 planner(numNodes, c, side_tids, aggGroups, joined);

  // run for a number of iterations
  planner.run_agg_first_only();

  // get the result of the planning
  auto result = planner.get_result();

  // write stuff to the page
  UseTemporaryAllocationBlock blk{page->getBytes(), page->getSize()};

  pdb::Handle<JoinPlannerResult> out = pdb::makeObject<JoinPlannerResult>();

  // this is the stuff we need to execute the query
  out->records0 = pdb::makeObject<pdb::Map<TRABlockMeta, int32_t>>();
  out->records1 = pdb::makeObject<pdb::Map<TRABlockMeta, int32_t>>();
  out->records2 = pdb::makeObject<pdb::Map<TRABlockMeta, int32_t>>();
  out->record_mapping = pdb::makeObject<pdb::Vector<bool>>(side_tids.size() * numNodes, side_tids.size() * numNodes);
  out->join_group_mapping = pdb::makeObject<pdb::Vector<int32_t>>(joined.size(), joined.size());
  out->joinedRecords = pdb::makeObject<pdb::Vector<Join3KeyPipeline::joined_record>>(joined.size(), joined.size());
  out->aggMapping = pdb::makeObject<pdb::Vector<int32_t>>(aggGroups.size(), aggGroups.size());
  out->aggRecords = pdb::makeObject<pdb::Vector< Handle< pdb::Vector<int32_t>> >>(aggGroups.size(), aggGroups.size());

  // zero out the record to node
  bzero(out->record_mapping->c_ptr(), sizeof(bool) * side_tids.size() * numNodes);

  // go through the result
  for(int jg = 0; jg < joined.size(); ++jg) {

    // get the node the join group is assigned to
    int32_t node = 0;
    while(!result.join_groups_to_node[jg * numNodes + node]) { node++; }

    // set the join_group_mapping
    (*out->join_group_mapping)[jg] = node;

    // get the joined record
    auto &record = joined[jg];

    // copy the stuff
    (*out->joinedRecords)[jg].first = record.first;
    (*out->joinedRecords)[jg].second = record.second;
    (*out->joinedRecords)[jg].third = record.third;

    std::cout << "Joining A : " << record.first << " B : " << record.second << " C : " << record.third << " TID : " << jg << " on "<< node <<'\n';

    // do the record to node join_group_mapping
    (*out->record_mapping)[record.first * numNodes + node] = true;
    (*out->record_mapping)[record.second * numNodes + node] = true;
    (*out->record_mapping)[record.third * numNodes + node] = true;
  }

  // copy the records for A with tid mappings
  for(auto &r : nodeRecords0) {
    std::cout << "For set A - rowID : " << r.first.rowID << " colID : " << r.first.colID << " tid : " << r.second.first << '\n';
    (*out->records0)[TRABlockMeta(r.first.rowID, r.first.colID)] = r.second.first;
  }

  // copy the records for B with tid mappings
  for(auto &r : nodeRecords1) {
    std::cout << "For set B - rowID : " << r.first.rowID << " colID : " << r.first.colID << " tid : " << r.second.first << '\n';
    (*out->records1)[TRABlockMeta(r.first.rowID, r.first.colID)] = r.second.first;
  }

  // copy the records for C with tid mappings
  for(auto &r : nodeRecords2) {
    std::cout << "For set C - rowID : " << r.first.rowID << " colID : " << r.first.colID << " tid : " << r.second.first << '\n';
    (*out->records2)[TRABlockMeta(r.first.rowID, r.first.colID)] = r.second.first;
  }

  // store the aggregation groups
  for(int32_t i = 0; i < aggGroups.size(); ++i) {

    // find the assigment
    int32_t node = 0;
    while(!result.agg_group_assignments[i * numNodes + node]) { node++; }

    // store it
    (*out->aggMapping)[i] = node;

    // store the aggregation groups
    (*out->aggRecords)[i] = pdb::makeObject< pdb::Vector<int32_t> >(aggGroups[i].size(), aggGroups[i].size());
    auto &toFill = *(*out->aggRecords)[i];
    for(auto j = 0; j < aggGroups[i].size(); ++j) {
      toFill[j] = aggGroups[i][j];
    }
  }

  // set the root object
  getRecord(out);
}