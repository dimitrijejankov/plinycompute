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
#ifndef LA_RUNING_ENVIRONMENT_H
#define LA_RUNING_ENVIRONMENT_H

#include <ctime>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <chrono>
#include <fcntl.h>
#include <set>
#include <map>

#include "PDBDebug.h"
#include "PDBString.h"
#include "Lambda.h"
#include "PDBClient.h"
#include "LADimension.h"

// by Binhang, June 2017

class LAPDBInstance {

private:
    bool print_result_;
    bool cluster_mode_;
    size_t block_size_;
    std::string manager_ip_;
    int port_;
    pdb::PDBClient pdb_client_;

    int dispatch_count_ = 0;

    std::set<std::string> cached_set_;
    std::map<std::string, std::string> identifier_pdb_set_name_map_;
    std::map<std::string, LADimension> identifier_dimension_map;

public:
    LAPDBInstance(bool print_result,
                  bool cluster_mode,
                  size_t block_size,
                  std::string manager_ip,
                  int port)
        : print_result_(print_result),
          cluster_mode_(cluster_mode),
          block_size_(block_size),
          manager_ip_(manager_ip),
          port_(port),
          pdb_client_(8108, manager_ip) {
        // Register libraries;
        pdb_client_.registerType("libraries/libLAAddJoin.so");
        pdb_client_.registerType("libraries/libLAColMaxAggregate.so");
        pdb_client_.registerType("libraries/libLAColMinAggregate.so");
        pdb_client_.registerType("libraries/libLAColSumAggregate.so");
        pdb_client_.registerType("libraries/libLADuplicateColMultiSelection.so");
        pdb_client_.registerType("libraries/libLADuplicateRowMultiSelection.so");
        pdb_client_.registerType("libraries/libLAElementwiseMultiplyJoin.so");
        pdb_client_.registerType("libraries/libLAInverse1Aggregate.so");
        pdb_client_.registerType("libraries/libLAInverse2Selection.so");
        pdb_client_.registerType("libraries/libLAInverse3MultiSelection.so");
        pdb_client_.registerType("libraries/libLAMaxElementAggregate.so");
        pdb_client_.registerType("libraries/libLAMaxElementOutputType.so");
        pdb_client_.registerType("libraries/libLAMaxElementValueType.so");
        pdb_client_.registerType("libraries/libLAMinElementAggregate.so");
        pdb_client_.registerType("libraries/libLAMinElementOutputType.so");
        pdb_client_.registerType("libraries/libLAMinElementValueType.so");
        pdb_client_.registerType("libraries/libLAMultiply1Join.so");
        pdb_client_.registerType("libraries/libLAMultiply2Aggregate.so");
        pdb_client_.registerType("libraries/libLARowMaxAggregate.so");
        pdb_client_.registerType("libraries/libLARowMinAggregate.so");
        pdb_client_.registerType("libraries/libLARowSumAggregate.so");
        pdb_client_.registerType("libraries/libLAScanMatrixBlockSet.so");
        pdb_client_.registerType("libraries/libLASingleMatrix.so");
        pdb_client_.registerType("libraries/libLASubtractJoin.so");
        pdb_client_.registerType("libraries/libLATransposeMultiply1Join.so");
        pdb_client_.registerType("libraries/libLATransposeSelection.so");
        pdb_client_.registerType("libraries/libLAWriteMatrixBlockSet.so");
        pdb_client_.registerType("libraries/libLAWriteMaxElementSet.so");
        pdb_client_.registerType("libraries/libLAWriteMinElementSet.so");
        pdb_client_.registerType("libraries/libMatrixBlock.so");
        pdb_client_.registerType("libraries/libMatrixData.so");
        pdb_client_.registerType("libraries/libMatrixMeta.so");

        pdb_client_.createDatabase("LA_db");
        std::cout << "Created database <LA_db>."<<std::endl;

    }


    pdb::PDBClient getPDBClient(){
        return this->pdb_client_;
    }

    void increaseDispatchCount() {
        this->dispatch_count_ += 1;
    }

    int getDispatchCount() {
        return this->dispatch_count_;
    }

    size_t getBlockSize() {
        return this->block_size_;
    }


    void addToCachedSet(std::string setName) {
        this->cached_set_.insert(setName);
    }

    void deleteFromCachedSet(std::string setName) {
        this->cached_set_.erase(setName);
    }

    bool existsPDBSet(std::string setName) {
        return this->cached_set_.find(setName) != this->cached_set_.end();
    }

    void addToIdentifierPDBSetNameMap(std::string identifierName, std::string scanSetName) {
        if (this->identifier_pdb_set_name_map_.find(identifierName) != this->identifier_pdb_set_name_map_.end()) {
            this->identifier_pdb_set_name_map_.erase(identifierName);
        }
        this->identifier_pdb_set_name_map_.insert(
            std::pair<std::string, std::string>(identifierName, scanSetName));
    }

    bool existsPDBSetForIdentifier(std::string identifierName) {
        return this->identifier_pdb_set_name_map_.find(identifierName) != this->identifier_pdb_set_name_map_.end();
    }

    std::string getPDBSetNameForIdentifier(std::string identifierName) {
        return this->identifier_pdb_set_name_map_[identifierName];
    }

    void addToIdentifierDimensionMap(std::string identifierName, LADimension dim) {
        if (this->identifier_dimension_map.find(identifierName) != this->identifier_dimension_map.end()) {
            this->identifier_dimension_map.erase(identifierName);
        }
        this->identifier_dimension_map.insert(std::pair<std::string, LADimension>(identifierName, dim));
    }

    bool existsDimension(std::string identifierName) {
        return this->identifier_dimension_map.find(identifierName) != this->identifier_dimension_map.end();
    }

    LADimension findDimension(std::string identifierName) {
        return this->identifier_dimension_map[identifierName];
    }

    void clearCachedSets() {
        std::cout << "Clear all the set in LA_db." << std::endl;
        for (auto const& setName : this->cached_set_) {
            this->pdb_client_.removeSet("LA_db", setName);
            std::cout << "Set " << setName << " removed" << std::endl;
        }
        this->cached_set_.clear();
    }
};

#endif
