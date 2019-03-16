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
#include <InputTupleSetSpecifier.h>


pdb::InputTupleSetSpecifier::InputTupleSetSpecifier() {
    this->tupleSetName = "";
}


pdb::InputTupleSetSpecifier::InputTupleSetSpecifier(std::string tupleSetName,
                                                    std::vector<std::string> columnNamesToKeep,
                                                    std::vector<std::string> columnNamesToApply) {
    this->tupleSetName = tupleSetName;
    for (int i = 0; i < columnNamesToKeep.size(); i++) {
        this->columnNamesToKeep.push_back(columnNamesToKeep[i]);
    }
    for (int i = 0; i < columnNamesToApply.size(); i++) {
        this->columnNamesToApply.push_back(columnNamesToApply[i]);
    }

}

pdb::InputTupleSetSpecifier::~InputTupleSetSpecifier() {}

std::string &pdb::InputTupleSetSpecifier::getTupleSetName() {
    return this->tupleSetName;
}

std::vector<std::string> &pdb::InputTupleSetSpecifier::getColumnNamesToKeep() {
    return this->columnNamesToKeep;
}

std::vector<std::string> &pdb::InputTupleSetSpecifier::getColumnNamesToApply() {
    return this->columnNamesToApply;
}

void pdb::InputTupleSetSpecifier::print() {

    std::cout << "TupleSetName:" << tupleSetName << std::endl;
    std::cout << "Columns to keep in output:" << std::endl;
    for (int i = 0; i < columnNamesToKeep.size(); i++) {
        std::cout << columnNamesToKeep[i] << std::endl;
    }
    std::cout << "Columns to apply:" << std::endl;
    for (int i = 0; i < columnNamesToApply.size(); i++) {
        std::cout << columnNamesToApply[i] << std::endl;
    }

}

void pdb::InputTupleSetSpecifier::clear() {
    this->columnNamesToKeep.clear();
    this->columnNamesToApply.clear();
}
