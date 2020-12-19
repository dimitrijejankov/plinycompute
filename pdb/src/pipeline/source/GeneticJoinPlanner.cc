#include <GeneticJoinPlanner.h>

pdb::GeneticJoinPlanner::GeneticJoinPlanner(int32_t numRecords,
                                            int32_t numJoinRecords,
                                            int32_t numNodes,
                                            int32_t sideRecordSize,
                                            std::vector<std::vector<bool>> &side_tids,
                                            std::vector<Join3KeyPipeline::joined_record> &joinedRecords,
                                            int32_t populationSize) : node_dist(0, numNodes - 1),
                                                                      selection_dist(0, (populationSize - 1) / 2),
                                                                      join_group_dist(0, numJoinRecords - 1),
                                                                      numRecords(numRecords),
                                                                      numJoinRecords(numJoinRecords),
                                                                      numNodes(numNodes),
                                                                      sideRecordSize(sideRecordSize),
                                                                      side_tids(side_tids),
                                                                      joinedRecords(joinedRecords),
                                                                      populationSize(populationSize) {
  tempRequirements.resize(numNodes);
  for(auto &t : tempRequirements) {
    t.resize(numRecords);
  }

  generator.seed(std::chrono::system_clock::now().time_since_epoch().count());
  tempStats.resize(numNodes);
}