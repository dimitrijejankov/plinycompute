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
#include "pipeline/Pipeline.h"


pdb::Pipeline::Pipeline(PDBAnonymousPageSetPtr outputPageSet, ComputeSourcePtr dataSource, ComputeSinkPtr tupleSink) :
        outputPageSet(std::move(outputPageSet)), dataSource(std::move(dataSource)), dataSink(std::move(tupleSink)) {}

pdb::Pipeline::~Pipeline() {

    // kill all of the pipeline stages
    while (!pipeline.empty())
        pipeline.pop_back();

    // first, reverse the queue so we go oldest to newest
    // this ensures that everything is deleted in the reverse order that it was created
    std::vector<MemoryHolderPtr> reverser;
    while (!unwrittenPages.empty()) {
        reverser.push_back(unwrittenPages.front());
        unwrittenPages.pop();
    }

    while (!reverser.empty()) {
        unwrittenPages.push(reverser.back());
        reverser.pop_back();
    }

    // write back all of the pages
    cleanPages(999999999);

    if (!unwrittenPages.empty())
        std::cout << "This is bad: in destructor for pipeline, still some pages with objects!!\n";

}

// adds a stage to the pipeline
void pdb::Pipeline::addStage(ComputeExecutorPtr addMe) {
    pipeline.push_back(addMe);
}

// writes back any unwritten pages
void pdb::Pipeline::cleanPages(int iteration) {

    // take care of getting rid of any pages... but only get rid of those from two iterations ago...
    // pages from the last iteration may still have pointers into them
    PDB_COUT << "to clean page for iteration-" << iteration << std::endl;
    PDB_COUT << "unwrittenPages.size() =" << unwrittenPages.size() << std::endl;

    while (!unwrittenPages.empty() && iteration > unwrittenPages.front()->iteration + 1) {

        PDB_COUT << "unwrittenPages.front()->iteration=" << unwrittenPages.front()->iteration << std::endl;

        // in this case, the page did not have any output data written to it... it only had
        // intermediate results, and so we will just discard it
        if (unwrittenPages.front()->outputSink == nullptr) {

            if (getNumObjectsInAllocatorBlock(unwrittenPages.front()->pageHandle->getBytes()) != 0) {

                // this is bad... there should not be any objects here because this memory
                // chunk does not store an output vector
                emptyOutContainingBlock(unwrittenPages.front()->pageHandle->getBytes());

                std::cout << "This is Strange... how did I find a page with objects??\n";
            }

            // remove the page from the output set
            outputPageSet->removePage(unwrittenPages.front()->pageHandle);
            unwrittenPages.pop();

            // in this case, the page DID have some data written to it
        } else {

            // and force the reference count for this guy to go to zero
            PDB_COUT << "to empty out containing block" << std::endl;
            unwrittenPages.front()->outputSink.emptyOutContainingBlock();

            // OK, because we will have invalidated the current object allocator block, we need to
            // create a new one, or this could cause a lot of problems!!
            if (iteration == 999999999) {
                makeObjectAllocatorBlock(1024, true);
            }

            // unpin the page so we don't have problems
            unwrittenPages.front()->pageHandle->unpin();

            // and get rid of him
            unwrittenPages.pop();
        }
    }
}

// runs the pipeline
void pdb::Pipeline::run() {

    // this is where we are outputting all of our results to
    MemoryHolderPtr myRAM = std::make_shared<MemoryHolder>(outputPageSet->getNewPage());

    // and here is the chunk
    TupleSetPtr curChunk;

    // the iteration counter
    int iteration = 0;

    // while there is still data
    while ((curChunk = dataSource->getNextTupleSet()) != nullptr) {

        // go through all of the pipeline stages
        for (ComputeExecutorPtr &q : pipeline) {

            try {
                curChunk = q->process(curChunk);

            } catch (NotEnoughSpace &n) {

                // and get a new page
                myRAM->setIteration(iteration);
                unwrittenPages.push(myRAM);
                myRAM = std::make_shared<MemoryHolder>(outputPageSet->getNewPage());

                // then try again
                curChunk = q->process(curChunk);
            }
        }

        try {

            if (myRAM->outputSink == nullptr) {
                myRAM->outputSink = dataSink->createNewOutputContainer();
            }

            dataSink->writeOut(curChunk, myRAM->outputSink);

        } catch (NotEnoughSpace &n) {

            // again, we ran out of RAM here, so write back the page and then create a new output page
            myRAM->setIteration(iteration);
            unwrittenPages.push(myRAM);
            myRAM = std::make_shared<MemoryHolder>(outputPageSet->getNewPage());

            // and again, try to write back the output
            myRAM->outputSink = dataSink->createNewOutputContainer();
            dataSink->writeOut(curChunk, myRAM->outputSink);
        }

        // lastly, write back all of the output pages
        iteration++;
        cleanPages(iteration);
    }

    // set the iteration
    myRAM->setIteration(iteration);

    // and remember the page
    unwrittenPages.push(myRAM);
}
