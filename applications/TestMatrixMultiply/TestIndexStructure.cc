#include <UseTemporaryAllocationBlock.h>
#include "sharedLibraries/headers/TensorBlock.h"
#include <unordered_map>

struct IndexNode {

  // if this is a leaf node it points to a std::vector<int32_t, int32_t> where each pair is <page_idx, record_idx>
  // otherwise it points to a map<int32_t, shared_ptr<IndexNode>>
  bool isLeaf = false;

  // points to the data of this node
  void *data;

  explicit IndexNode(bool isLeaf) : isLeaf(isLeaf) {

    if(isLeaf) {
      data = new std::vector<std::pair<int32_t, int32_t>>();
    }
    else  {
      data = new std::map<int32_t, std::shared_ptr<IndexNode>>();
    }
  }

  ~IndexNode() {
    if(isLeaf) {
      delete ((std::vector<std::pair<int32_t, int32_t>>*) data);
    }
    else {
      delete ((std::map<int32_t, std::shared_ptr<IndexNode>>*) data);
    }
  }

  void get(std::vector<std::pair<int32_t, int32_t>> &out, const std::vector<int32_t> &index) {
    _get(out, index, 0);
  }


  void insert(pdb::matrix::TensorBlockMeta &block, const std::pair<int32_t, int32_t> &location) {

    _insert(block, 0, location);
  }

private:

  void _get(std::vector<std::pair<int32_t, int32_t>> &out, const std::vector<int32_t> &index, int depth) {

    // if this is a leaf store them
    if(isLeaf) {

      // insert all the indices
      auto &d = *((std::vector<std::pair<int32_t, int32_t>>*) data);
      out.insert(out.end(), d.begin(), d.end());
    }
    else {

      // get the index
      auto idx = index[depth];

      // get the map
      auto &d = *((std::map<int32_t, std::shared_ptr<IndexNode>>*) data);

      if(idx != -1) {

        // find it and go one step deeper
        auto it = d.find(idx);
        if(it != d.end()) {
          it->second->_get(out, index, depth + 1);
        }
      }
      else {

        // go one level deeper for ech of them
        for(auto &it : d) {
          it.second->_get(out, index, depth + 1);
        }
      }
    }
  }

  void _insert(pdb::matrix::TensorBlockMeta &block, int depth, const std::pair<int32_t, int32_t> &location) {

    if(isLeaf) {

      // cast the data and store the record location
      ((std::vector<std::pair<int32_t, int32_t>>*) data)->emplace_back(location);
    }
    else {

      // get the index
      auto index = block.indices[depth];

      // get the map
      auto &d = *((std::map<int32_t, std::shared_ptr<IndexNode>>*) data);

      // try to find the index
      auto it = d.find(index);
      if(it == d.end()) {

        // insert the index node
        auto tmp = d.insert({index, std::make_shared<IndexNode>((block.indices.size() - 1) == depth)});
        it = tmp.first;
      }

      // insert into the new node
      it->second->_insert(block, depth + 1, location);
    }
  }

};

int main() {

  // use temporary allocation block
  const pdb::UseTemporaryAllocationBlock tempBlock{128 * 1024 * 1024};

  std::vector<pdb::Handle<pdb::matrix::TensorBlock>> blocks;
  for(int i = 0; i < 10; i++) {
    for(int j = 0; j < 10; j++) {
      for(int k = 0; k < 10; k++) {
        blocks.emplace_back(pdb::makeObject<pdb::matrix::TensorBlock>(i, j, k, 10, 10, 10));
      }
    }
  }

  // insert them
  auto root = std::make_shared<IndexNode>(false);
  for(auto &block : blocks) {
    root->insert(*block->metaData, {0, 0});
  }

  std::vector<std::pair<int32_t, int32_t>> out;
  root->get(out, {6, -1, 6});

  std::cout << out.size() << '\n';
  return 0;
}