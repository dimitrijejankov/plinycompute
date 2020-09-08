#include <UseTemporaryAllocationBlock.h>
#include "TRABlock.h"
#include "TRAIndex.h"
#include <unordered_map>

int main() {

  // use temporary allocation block
  const pdb::UseTemporaryAllocationBlock tempBlock{128 * 1024 * 1024};

  std::vector<pdb::Handle<pdb::TRABlock>> blocks;
  for(int i = 0; i < 10; i++) {
    for(int j = 0; j < 10; j++) {
      for(int k = 0; k < 10; k++) {
        blocks.emplace_back(pdb::makeObject<pdb::TRABlock>(i, j, k, 10, 10, 10));
      }
    }
  }

  // insert them
  auto root = std::make_shared<pdb::TRAIndexNode>(false);
  for(auto &block : blocks) {
    root->insert(*block->metaData, {0, 0});
  }

  std::vector<std::pair<int32_t, int32_t>> out;
  root->get(out, {6, -1, 6});

  std::cout << out.size() << '\n';
  return 0;
}