#pragma once

//  PRELOAD %TRABlock%

#include <Object.h>
#include <Handle.h>
#include "TRABlockMeta.h"
#include "TRABlockData.h"

namespace pdb {

/**
* This represents a block in a large matrix distributed matrix.
* For example if the large matrix has the size of 10000x10000 and is split into 4 blocks of size 2500x2500
* Then we would have the following blocks in the system
*
* |metaData.key1|metaData.key0|data.numRows|data.numCols| data.block |
* |       0      |       1      |    25k     |    25k     | 25k * 25k  |
* |       1      |       1      |    25k     |    25k     | 25k * 25k  |
* |       0      |       0      |    25k     |    25k     | 25k * 25k  |
* |       1      |       0      |    25k     |    25k     | 25k * 25k  |
*/
class TRABlock : public pdb::Object {
public:

    /**
    * The default constructor
    */
    TRABlock() = default;

    /**
    * The constructor for a block size
    * @param rowID - the value we want to initialize the row id to
    * @param colID - the value we want to initialize the col id to
    * @param numRows - the number of rows the block has
    * @param numCols - the number of columns the block has
    */
    TRABlock(uint32_t key0, uint32_t key1, uint32_t key2, uint32_t dim0, uint32_t dim1, uint32_t dim2) {
        metaData = makeObject<TRABlockMeta>(key0, key1, key2);
        data = makeObject<TRABlockData>(dim0, dim1, dim2);
    }

  TRABlock(uint32_t key0, uint32_t key1, uint32_t dim0, uint32_t dim1) {
    metaData = makeObject<TRABlockMeta>(key0, key1);
    data = makeObject<TRABlockData>(dim0, dim1, 1);
  }

    //Todo Binhang: we should check wil this work? Here I guess it will shuffle less amount of data by copy handle(pointer)
    TRABlock(uint32_t key0, uint32_t key1, uint32_t key2, Handle<TRABlockData>& existData) {
        metaData = makeObject<TRABlockMeta>(key0, key1, key2);
        data = existData;
    }

    ENABLE_DEEP_COPY

    /**
    * The metadata of the matrix
    */
    Handle<TRABlockMeta> metaData;

    /**
    * The data of the matrix
    */
    Handle<TRABlockData> data;

    /**
    *
    * @return
    */
    Handle<TRABlockMeta>& getKey() {
        return metaData;
    }

    /**
    *
    * @return
    */
    TRABlockMeta& getKeyRef(){
        return *metaData;
    }

    /**
    *
    * @return
    */
    Handle<TRABlockData>& getValue() {
        return data;
    }

    TRABlockData& getValueRef() {
        return *data;
    }

    uint32_t getkey0() {
        return metaData->getIdx0();
    }

    uint32_t getkey1() {
        return metaData->getIdx1();
    }

    uint32_t getkey2() {
        return metaData->getIdx2();
    }

    uint32_t getDim0() {
        return data->dim0;
    }

    uint32_t getDim1() {
        return data->dim1;
    }

    uint32_t getDim2() {
        return data->dim2;
    }

    void print_meta() {
      std::cout << "Meta : ";
      for(int i = 0; i < metaData->indices.size(); ++i) {
        std::cout << metaData->indices[i] << " ";
      }
      std::cout << '\n' << std::flush ;
    }

    void print_block() {
        data->print();
    }

    void print() {
        print_meta();
        print_block();
    }
};

}