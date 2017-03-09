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

#ifndef SCAN_USER_SET_H
#define SCAN_USER_SET_H

//by Jia, Mar 2017

#include "Computation.h"
#include "PageCircularBufferIterator.h"
#include "VectorTupleSetIterator.h"
#include "PDBString.h"
#include "DataTypes.h"
#include "DataProxy.h"
#include "Configuration.h"

namespace pdb {

template <class OutputClass>
class ScanUserSet : public Computation {

public:

        ENABLE_DEEP_COPY

        void initialize() {
            this->iterator = nullptr;
            this->proxy = nullptr;
        }

        ComputeSourcePtr getComputeSource (TupleSpec &schema, ComputePlan &plan) override {
             return std :: make_shared <VectorTupleSetIterator> (

                 [&] () -> void * {
                     if (this->iterator == nullptr) {
                         return nullptr;
                     }
                     while (this->iterator->hasNext() == true) {

                        PDBPagePtr page = this->iterator->next();
                        if(page != nullptr) {
                            return page->getBytes();
                        }
                     }
                     
                     return nullptr;

                 },

                 [&] (void * freeMe) -> void {
                     if (this->proxy != nullptr) {
                         char * pageRawBytes = (char *)freeMe-(sizeof(NodeID) + sizeof(DatabaseID) + sizeof(UserTypeID) + sizeof(SetID) + sizeof(PageID));
                         char * curBytes = pageRawBytes;
                         NodeID nodeId = (NodeID) (*((NodeID *)(curBytes)));
                         curBytes = curBytes + sizeof(NodeID);
                         DatabaseID dbId = (DatabaseID) (*((DatabaseID *)(curBytes)));
                         curBytes = curBytes + sizeof(DatabaseID);
                         UserTypeID typeId = (UserTypeID) (*((UserTypeID *)(curBytes)));
                         curBytes = curBytes + sizeof(UserTypeID);
                         SetID setId = (SetID) (*((SetID *)(curBytes)));
                         curBytes = curBytes + sizeof(SetID);
                         PageID pageId = (PageID) (*((PageID *)(curBytes)));
                         PDBPagePtr page = make_shared<PDBPage>(pageRawBytes, nodeId, dbId, typeId, setId, pageId, DEFAULT_PAGE_SIZE, 0, 0);
                         this->proxy->unpinUserPage (nodeId, dbId, typeId, setId, page);
                     }
                 },

                 this->batchSize

            );
        }
         
        //JiaNote: be careful here that we put PageCircularBufferIteratorPtr and DataProxyPtr in a pdb object
        void setIterator(PageCircularBufferIteratorPtr iterator) {
                this->iterator = iterator;
        }

        void setProxy(DataProxyPtr proxy) {
                this->proxy = proxy;
        }


        void setBatchSize(int batchSize) {
                this->batchSize = batchSize;

        }

        void setDatabaseName (std :: string dbName) {
                this->dbName = dbName;
        }

        void setSetName (std :: string setName) {
                this->setName = setName;
        }

        std :: string getDatabaseName () {
                return dbName;
        }

        std :: string getSetName () {
                return setName;
        }


	std :: string getComputationType () override {
		return std :: string ("ScanUserSet");
	}

protected:

       //JiaNote: be careful here that we put PageCircularBufferIteratorPtr and DataProxyPtr in a pdb object.
       PageCircularBufferIteratorPtr iterator;

       DataProxyPtr proxy;

       String dbName;
 
       String setName;

       int batchSize;

};




}

#endif
