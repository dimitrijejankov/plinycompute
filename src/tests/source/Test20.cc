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

#ifndef TEST_20_CC
#define TEST_20_CC

#include "PangeaStorageServer.h"
#include "PDBVector.h"
#include "Employee.h"
#include "InterfaceFunctions.h"
#include "UseTemporaryAllocationBlock.h"
#include "VTableMap.h"
#include <unistd.h>
#include <stdio.h>

int main (int argc, char * argv[]) {


        ConfigurationPtr conf = make_shared < Configuration > ();
        pdb :: PDBLoggerPtr logger = make_shared < pdb :: PDBLogger> (conf->getLogFile());
        SharedMemPtr shm = make_shared< SharedMem > (conf->getShmSize(), logger);
        pdb :: PDBWorkerQueuePtr workers = make_shared < pdb :: PDBWorkerQueue > (logger, conf->getMaxConnections()); 
        pdb :: PangeaStorageServerPtr storage = make_shared<pdb :: PangeaStorageServer> (shm, workers, logger, conf);	
        storage->startFlushConsumerThreads();
        
        //load set
        SetPtr set = storage->getSet(std :: pair <std :: string, std :: string> ("testDatabase", "testSet"));
        if( set == nullptr ) {
             std :: cout << "Set data hasn't been loaded! You must run Test19 first!" << std :: endl;
             exit (EXIT_FAILURE);
        } else {
             std :: cout << "Set data has been loaded!" << std :: endl;
        }
        storage->getCache()->pin(set, MRU, Read);



        /**
         * Frontend getIterators() API returns fixed number of iterators: NumOfDiskFilePartitions+1(In-memoryIterator);
         * Backend getIterators(int numThreads) API returns flexible number of iterators that can be specified by user;
         * Here we test frontend iterators.
         */
                
        vector<PageIteratorPtr> * iterators = set->getIterators();
        int numIterators = iterators->size();
        for (int i = 0; i < numIterators; i++) {
            PageIteratorPtr iter = iterators->at(i);
            while (iter->hasNext()){
                PDBPagePtr page = iter->next();
                if (page != nullptr) {
                    std :: cout << "processing page with pageId=" << page->getPageID()<<std :: endl;
                    auto * temp = (pdb :: Record<pdb :: Vector<pdb :: Handle<pdb :: Employee>>> *) page->getBytes();
                    auto employees = temp->getRootObject ();
                    for (int j = 0; j < employees->size(); j++) {
                       if (j % 10000 == 0) {
                           std :: cout << "the "<<j<<"-th employee in the "<<page->getPageID()<<"-th page:"<<std :: endl;
                           (*employees)[j]->print();
                           std :: cout << std :: endl;
                       }
                    }
                 }
                page->unpin();
            }
        }
                

        //let's finish
        std :: cout << "finish!" << std :: endl;
        storage->stopFlushConsumerThreads();

}

#endif

