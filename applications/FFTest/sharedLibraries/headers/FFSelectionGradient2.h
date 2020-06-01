/*****************************************************************************
 *                                                                           *
 *  Copyright 2018 Rice University                                           *
 *                                                                           *
 *  Licensed under the Apache License, Version 2.0 (the "License");          *
 *  you may not use this file except in compliance with the License.         *
 *  You may obtain a copy of the License at                                  *
 *                                                                           *
 *      http://www.apache.org/licenses/LICENSE-2.0                           *
 *                                                                           *
 *  Unless required by applicable law or agreed to in writing, software      *
 *  distributed under the License is distributed on an "AS IS" BASIS,        *
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. *
 *  See the License for the specific language governing permissions and      *
 *  limitations under the License.                                           *
 *                                                                           *
 *****************************************************************************/
#pragma once

#include "DoubleVector.h"
#include "Lambda.h"
#include "LambdaCreationFunctions.h"
#include "SelectionComp.h"
#include "FFMatrixBlock.h"
#include <cstdlib>
#include <ctime>

namespace pdb {

// the sub namespace
namespace ff {

// FFSelectionGradient2 will
class FFSelectionGradient2 : public SelectionComp<FFMatrixBlock, FFMatrixBlock> {

public:

  ENABLE_DEEP_COPY

  FFSelectionGradient2() = default;
  FFSelectionGradient2(int32_t num_row_blocks,
                       int32_t num_col_blocks,
                       const std::vector<std::pair<int32_t, int32_t>> &labels_meta,
                       const std::vector<std::pair<int32_t, int32_t>> &labels_data) : num_row_blocks(num_row_blocks),
                                                                                      num_col_blocks(num_col_blocks),
                                                                                      labels_meta_pos(labels_meta.size(), labels_meta.size()),
                                                                                      labels_meta_cnt(labels_meta.size(), labels_meta.size()),
                                                                                      labels_data_col(labels_data.size(), labels_data.size()),
                                                                                      labels_data_row(labels_data.size(), labels_data.size()) {

    // copy the meta
    for(int i = 0; i < labels_meta.size(); ++i) {
      this->labels_meta_pos[i] = labels_meta[i].first;
      this->labels_meta_cnt[i] = labels_meta[i].second;
    }

    // copy the data
    for(int i = 0; i < labels_data.size(); ++i) {
      this->labels_data_col[i] = labels_data[i].first;
      this->labels_data_row[i] = labels_data[i].second;
    }
  }

  // srand has already been invoked in server
  Lambda<bool> getSelection(Handle<FFMatrixBlock> checkMe) override {
    return makeLambda(checkMe, [&](Handle<FFMatrixBlock> &checkMe) { return true; });
  }

  Lambda<Handle<FFMatrixBlock>> getProjection(Handle<FFMatrixBlock> checkMe) override {
    return makeLambda(checkMe, [&](Handle<FFMatrixBlock> &checkMe) {


      // make an output
      Handle<FFMatrixBlock> out = makeObject<FFMatrixBlock>(checkMe->getRowID(),
                                                            checkMe->getColID(),
                                                            checkMe->getNumRows() ,
                                                            checkMe->getNumCols());

      // do the sigmoid function
      auto data = checkMe->data->data->c_ptr();
      auto outData = out->data->data->c_ptr();
      for(int32_t i = 0; i < checkMe->getNumRows() * checkMe->getNumCols(); i++) {
        outData[i] = 1 / (1 + exp(-data[i]));
      }

      // do the stuff we need to
      auto &meta_pos = labels_meta_pos[checkMe->getRowID() * num_col_blocks + checkMe->getColID()];
      auto &meta_cnt = labels_meta_cnt[checkMe->getRowID() * num_col_blocks + checkMe->getColID()];
      for(int idx = 0; idx < meta_cnt; idx++) {
        auto pos_row = labels_data_row[meta_pos + idx];
        auto pos_col = labels_data_col[meta_pos + idx];
        outData[pos_row * checkMe->getNumCols() + pos_col] -= 1;
      }

      return out;
    });
  }

  // how many rows does the whole matrix have
  int32_t num_row_blocks{};
  int32_t num_col_blocks{};

  // the labels we use to calculate the gradient
  pdb::Vector<int32_t> labels_meta_pos;
  pdb::Vector<int32_t> labels_meta_cnt;
  pdb::Vector<int32_t> labels_data_col;
  pdb::Vector<int32_t> labels_data_row;
};
}
}