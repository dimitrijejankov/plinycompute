#pragma once

#include <PDBPhysicalAlgorithm.h>

namespace pdb {

class TRALocalAggregationStage : public PDBPhysicalAlgorithmStage {
 public:

  struct AggregationIndex {

    // if this is a leaf node it points to a std::vector<int32_t, int32_t> where each pair is <page_idx, record_idx>
    // otherwise it points to a map<int32_t, shared_ptr<IndexNode>>
    bool isLeaf = false;

    // points to the data of this node
    std::map<int32_t, AggregationIndex> idx;

    std::pair<int32_t, int32_t> loc = {-1, -1};

    void insert(const std::vector<int32_t> &key, const std::pair<int32_t, int32_t> &location) {
      _insert(key, location, 0);
    }

    void _insert(const std::vector<int32_t> &key, const std::pair<int32_t, int32_t> &location, int32_t depth) {

      if (key.size() == depth) {

        // make this a leaf
        isLeaf = true;
        loc = location;

      } else {

        // forward the insert
        idx[key[depth]]._insert(key, location, depth + 1);
      }
    }

    const std::pair<int32_t, int32_t> & get(const std::vector<int32_t> &key) {
      return _get(key, 0);
    }

    const std::pair<int32_t, int32_t> & _get(const std::vector<int32_t> &key, int32_t depth) {

      if(isLeaf) {
        return loc;
      }

      auto it = idx.find(key[depth]);
      if(it == idx.end()) {
        return loc;
      }

      return idx[key[depth]]._get(key, depth + 1);
    }
  };

  TRALocalAggregationStage(const pdb::String &inputPageSet,
                           const pdb::Vector<int32_t>& indices,
                           const pdb::String &sink);

  bool setup(const Handle<pdb::ExJob> &job,
             const PDBPhysicalAlgorithmStatePtr &state,
             const std::shared_ptr<pdb::PDBStorageManagerBackend> &storage,
             const std::string &error) override;

  bool run(const Handle<pdb::ExJob> &job,
           const PDBPhysicalAlgorithmStatePtr &state,
           const std::shared_ptr<pdb::PDBStorageManagerBackend> &storage,
           const std::string &error) override;

  void cleanup(const pdb::PDBPhysicalAlgorithmStatePtr &state, const std::shared_ptr<pdb::PDBStorageManagerBackend> &storage) override;

  // source db
  pdb::String db;

  // source set
  pdb::String set;

  // the page
  const pdb::String &inputPageSet;

  // sink
  const pdb::String &sink;

  const pdb::Vector<int32_t>& indices;

  const static PDBSinkPageSetSpec *_sink;
  const static Vector<PDBSourceSpec> *_sources;
  const static String *_finalTupleSet;
  const static Vector<pdb::Handle<PDBSourcePageSetSpec>> *_secondarySources;
  const static Vector<PDBSetObject> *_setsToMaterialize;
};

}