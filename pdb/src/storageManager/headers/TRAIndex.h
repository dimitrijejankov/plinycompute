#pragma once

#include <TRABlockMeta.h>
#include <cassert>
namespace pdb {

struct TRAIndexNode {

  // if this is a leaf node it points to a std::vector<int32_t, int32_t> where each pair is <page_idx, record_idx>
  // otherwise it points to a map<int32_t, shared_ptr<IndexNode>>
  bool isLeaf = false;

  // points to the data of this node
  void *data;

  explicit TRAIndexNode(bool isLeaf) : isLeaf(isLeaf) {

    if(isLeaf) {
      data = new std::vector<std::pair<int32_t, int32_t>>();
    }
    else  {
      data = new std::map<int32_t, std::shared_ptr<TRAIndexNode>>();
    }
  }

  ~TRAIndexNode() {
    if(isLeaf) {
      delete ((std::vector<std::pair<int32_t, int32_t>>*) data);
    }
    else {
      delete ((std::map<int32_t, std::shared_ptr<TRAIndexNode>>*) data);
    }
  }

  void get(std::vector<std::pair<int32_t, int32_t>> &out, const std::vector<int32_t> &index) {
    _get(out, index, 0);
  }


  void insert(pdb::TRABlockMeta &block, const std::pair<int32_t, int32_t> &location) {

    _insert(block, 0, location);
  }

  // needs to accept a vector with -1 at least on the first usage...
  // will move this
  static void getRHSMatcher(const pdb::Vector<uint32_t> &lhs_index,
                                            const std::vector<int32_t>& lhs_indices,
                                            const std::vector<int32_t>& rhs_indices,
                                            std::vector<int32_t> &rhsMatch) {
    assert(lhs_indices.size() == rhs_indices.size());

    for(int i = 0; i < lhs_indices.size(); ++i) {
      rhsMatch[rhs_indices[i]] = lhs_index[lhs_indices[i]];
    }
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
      auto &d = *((std::map<int32_t, std::shared_ptr<TRAIndexNode>>*) data);

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

  void _insert(pdb::TRABlockMeta &block, int depth, const std::pair<int32_t, int32_t> &location) {

    if(isLeaf) {

      // cast the data and store the record location
      ((std::vector<std::pair<int32_t, int32_t>>*) data)->emplace_back(location);
    }
    else {

      // get the index
      auto index = block.indices[depth];

      // get the map
      auto &d = *((std::map<int32_t, std::shared_ptr<TRAIndexNode>>*) data);

      // try to find the index
      auto it = d.find(index);
      if(it == d.end()) {

        // insert the index node
        auto tmp = d.insert({index, std::make_shared<TRAIndexNode>((block.indices.size() - 1) == depth)});
        it = tmp.first;
      }

      // insert into the new node
      it->second->_insert(block, depth + 1, location);
    }
  }

};

using TRAIndexNodePtr = std::shared_ptr<TRAIndexNode>;

}