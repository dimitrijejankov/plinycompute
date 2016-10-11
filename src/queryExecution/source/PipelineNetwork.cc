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
#ifndef PIPELINE_NETWORK_CC
#define PIPELINE_NETWORK_CC

//by Jia, Sept 2016

#include "PipelineNetwork.h"
#include "PageCircularBufferIterator.h"
#include "DataProxy.h"
#include "PageScanner.h"
#include "PageCircularBufferIterator.h"
#include "BlockQueryProcessor.h"
#include "InterfaceFunctions.h"
#include "HermesExecutionServer.h"
#include "GenericWork.h"
#include "SingleTableBundleProcessor.h"

namespace pdb {

PipelineNetwork :: ~PipelineNetwork () {
    std :: cout << "PipelineNetwork destructor is running" << std :: endl;
    sourceNodes->clear();
    delete sourceNodes;
    allNodes->clear();
    delete allNodes;
    this->jobStage = nullptr;
}

PipelineNetwork :: PipelineNetwork (SharedMemPtr shm, PDBLoggerPtr logger, ConfigurationPtr conf, NodeID nodeId, size_t batchSize, int numThreads) {
    sourceNodes = new std :: vector<PipelineNodePtr> ();
    allNodes = new std :: unordered_map<OperatorID, PipelineNodePtr>();
    this->batchSize = batchSize;
    this->numThreads = numThreads;
    this->nodeId = nodeId;
    this->logger = logger;
    this->conf = conf;
    this->shm = shm;
    this->id = 0;
}

//initialize a linear pipeline network
void PipelineNetwork :: initialize (Handle<JobStage> stage) {
    this->jobStage = stage;
    Vector<Handle<ExecutionOperator>> operators = stage->getOperators();
    bool isSource;
    bool isSink;
    Handle<SetIdentifier> input;
    Handle<SetIdentifier> output;
    PipelineNodePtr parentNode = nullptr;
    for (int i = 0; i < operators.size(); i ++) {
        isSource = false;
        isSink = false;
        input = nullptr;
        output = nullptr;
        if (i == 0) {
            isSource = true;
            input = stage->getInput();
        }
        if (i == operators.size()-1) {
            isSink = true;
            output = stage->getOutput();
        } 
        PipelineNodePtr node = make_shared<PipelineNode>(this->nodeId, operators[i], isSource, isSink, input, output, id);
        if (i == 0) {
            std :: cout << "to append source node with id=" << id << std :: endl;
            appendSourceNode (node);
        } else {
            std :: cout << "to append child node with id=" << id << " to parent with id=" << parentNode->getOperatorId() << std :: endl;
            appendNode (parentNode->getOperatorId(), node);
        }
        id ++;
        parentNode = node;
    }
}


//initialize a tree pipeline network
void PipelineNetwork :: initialize (PipelineNodePtr parentNode, Handle<JobStage> stage) {
    if(parentNode == nullptr) {
        this->jobStage = stage; 
    }
    Handle<JobStage> curStage = stage;
    if (curStage != nullptr) {   
        PipelineNodePtr nextParentNode = nullptr;
        Vector<Handle<ExecutionOperator>> operators = stage->getOperators();
        Handle<ExecutionOperator> sourceOperator = operators[0];
        bool isSink;
        Handle<SetIdentifier> outputSet = nullptr;
        if (operators.size() == 1) {
            isSink = true;
            outputSet = stage->getOutput();
        } else {
            isSink = false;
        }
        PipelineNodePtr node = nullptr;
        if (parentNode == nullptr) {
            Handle<SetIdentifier> inputSet = stage->getInput();
            node = make_shared<PipelineNode>(this->nodeId, operators[0], true, isSink, inputSet, outputSet, id);
            appendSourceNode (node);
        } else {
            node = make_shared<PipelineNode>(this->nodeId, operators[0], false, isSink, nullptr, outputSet, id);
            appendNode(parentNode->getOperatorId(), node);
        }
        if (isSink == true) {
            nextParentNode = node;
        }
        id ++;
        for (int i = 1; i < operators.size(); i++) {
            Handle<ExecutionOperator> curOperator = operators[i];
            bool isSource = false;
            Handle<SetIdentifier> inputSet = nullptr;
            bool isSink = false;
            Handle<SetIdentifier> outputSet = nullptr;
            if(i == operators.size()-1) {
                isSink = true;
                outputSet = stage->getOutput();
            }
            PipelineNodePtr node = make_shared<PipelineNode>(this->nodeId, curOperator, isSource, isSink, inputSet, outputSet, id);
            appendNode(id - 1, node);
            id ++;
            if (isSink == true) {
                nextParentNode = node;
            }
        }
        Vector<Handle<JobStage>> childrenStage = stage->getChildrenStages();
        for (int i = 0; i < childrenStage.size(); i ++) {
            initialize(nextParentNode, childrenStage[i]);
        }
    }
}


Handle<JobStage> & PipelineNetwork :: getJobStage () {
    return jobStage;
}

std :: vector<PipelineNodePtr> * PipelineNetwork :: getSourceNodes() {
    return this->sourceNodes;
}

bool PipelineNetwork :: appendNode (OperatorID parentId, PipelineNodePtr node) {
    auto iter = allNodes->find(parentId);
    if ( iter == allNodes->end() ) {
        std :: cout << "Can't find parent node" << std :: endl;
        return false;
    } else {
        (*allNodes)[parentId]->addChild(node);
        (*allNodes)[node->getOperatorId()] = node;
        return true;
    }
}

void PipelineNetwork :: appendSourceNode (PipelineNodePtr node) {
    this->sourceNodes->push_back(node);
    (*allNodes)[node->getOperatorId()] = node;
    return;
}

int PipelineNetwork :: getNumThreads () {
    return this->numThreads;

}

int PipelineNetwork :: getNumSources () {
     return this->sourceNodes->size();
} 

void PipelineNetwork :: runAllSources() {
     //TODO
}

void PipelineNetwork :: runSource (int sourceNode, HermesExecutionServer * server) {
    std :: cout << "Pipeline network is running" << std :: endl;
    bool success;
    std :: string errMsg;

    PipelineNodePtr source = this->sourceNodes->at(sourceNode);
    //initialize the data proxy, scanner and set iterators
    PDBCommunicatorPtr communicatorToFrontend = make_shared<PDBCommunicator>(); 
    communicatorToFrontend->connectToInternetServer(logger, conf->getPort(), "localhost", errMsg);

    //getScanner
    int backendCircularBufferSize = 3;
    PageScannerPtr scanner = make_shared<PageScanner>(communicatorToFrontend, shm, logger, numThreads, backendCircularBufferSize, nodeId); 
    if (server->getFunctionality<HermesExecutionServer>().setCurPageScanner(scanner) == false) {
        success = false;
        errMsg = "Error: A job is already running!";
        std :: cout << errMsg << std :: endl;
        return;
    }

    //get input set information
    Handle<SetIdentifier> inputSet = source->getInputSet();

    //get iterators
    //TODO: we should get iterators using only databaseName and setName
    std :: cout << "To send GetSetPages message" << std :: endl;
    std :: vector<PageCircularBufferIteratorPtr> iterators = scanner->getSetIterators(nodeId, inputSet->getDatabaseId(), inputSet->getTypeId(), inputSet->getSetId());
    std :: cout << "GetSetPages message is sent" << std :: endl;
    int numIteratorsReturned = iterators.size();
    if (numIteratorsReturned != numThreads) {
         success = false;
         errMsg = "Error: number of iterators doesn't match number of threads!";
         std :: cout << errMsg << std :: endl;
         return;
    }   

    //get output set information
    //now due to limitation in object model, we only support one output for a pipeline network
    Handle<SetIdentifier> outputSet = this->jobStage->getOutput();
    
    

    //create a buzzer and counter
    PDBBuzzerPtr tempBuzzer = make_shared<PDBBuzzer>(
         [&] (PDBAlarm myAlarm, int & counter) {
             counter ++;
             std :: cout << "counter = " << counter << std :: endl;
         });
    
    int counter = 0;
    int batchSize = 100;
    for (int i = 0; i < numThreads; i++) {
         PDBWorkerPtr worker = server->getFunctionality<HermesExecutionServer>().getWorkers()->getWorker();
         std :: cout << "to run the " << i << "-th work..." << std :: endl;
         //TODO: start threads
         PDBWorkPtr myWork = make_shared<GenericWork> (
             [&, i] (PDBBuzzerPtr callerBuzzer) {
                  //create a data proxy
                  PDBCommunicatorPtr anotherCommunicatorToFrontend = make_shared<PDBCommunicator>();
                  anotherCommunicatorToFrontend->connectToInternetServer(logger, conf->getPort(), "localhost", errMsg);
                  DataProxyPtr proxy = make_shared<DataProxy>(nodeId, anotherCommunicatorToFrontend, shm, logger);

                  //setup an output page to store intermediate results and final output
                  PDBPagePtr output;
                  proxy->addUserPage(outputSet->getDatabaseId(), outputSet->getTypeId(), outputSet->getSetId(), output);
                  std :: cout << "pinned page in output set with id=" << output->getPageID() << std :: endl;
                  makeObjectAllocatorBlock (output->getSize(), true);
                  Handle<Vector<Handle<Object>>> outputVec = makeObject<Vector<Handle<Object>>>();
             
                  //setup pipeline context
                  PipelineContextPtr context = make_shared<PipelineContext>(outputVec, proxy, outputSet);
                  context->setPageToUnpin(output);
 
                  //create a bundle processor
                  SingleTableBundleProcessorPtr bundler = make_shared<SingleTableBundleProcessor>();
                  bundler->setContext(context);
                  bundler->initialize();

                  PageCircularBufferIteratorPtr iter = iterators.at(i);
                  while (iter->hasNext()) {
                      std :: cout << "got a page!" << std :: endl;
                      PDBPagePtr page = iter->next();
                      if (page != nullptr) {
                          std :: cout << "page is not null with pageId="<< page->getPageID() << std :: endl;
                          bundler->loadInputPage(page->getBytes());
                          std :: cout << "load an allocate block for output" << std :: endl;
                          Handle<GenericBlock> outputBlock;
                          try {
                              outputBlock = bundler->loadOutputBlock(batchSize);
                          }
                          catch (NotEnoughSpace &n) {
                              std :: cout << "Error: allocator block size should be larger than vector initilaization size" << std :: endl;
                              exit(-1);
                          }
                          while(bundler->fillNextOutputBlock()) {
                              //std :: cout << "written one block!" << std :: endl;
                              if (context->isOutputFull()) {
                                  std :: cout << "current block is full, copy to output page!" << std :: endl;
                                  Record<Vector<Handle<Object>>> * myBytes = getRecord(context->getOutputVec());
                                  memcpy(context->getPageToUnpin()->getBytes(), myBytes, myBytes->numBytes());
                                  context->setNeedUnpin(true);
                                  proxy->addUserPage(outputSet->getDatabaseId(), outputSet->getTypeId(), outputSet->getSetId(), output);
                                  std :: cout << "pinned page in output set with id=" << output->getPageID() << std :: endl;
                                  makeObjectAllocatorBlock (output->getSize(), true);
                                  outputVec = makeObject<Vector<Handle<Object>>>();
                                  context->setOutputVec(outputVec);
                                  context->setOutputFull(false);
                              }

                              //we assume a run of pipeline will not consume all memory that has just allocated
                              //std :: cout << "run the pipeline on this block" << std :: endl;
                              source->run(context, outputBlock, batchSize);
                              bundler->clearOutputBlock();

                              //now we can unpin the previous page
                              if (context->shallWeUnpin()) {
                                  std :: cout << "we need to unpin the full page" << std :: endl;
                                  proxy->unpinUserPage(nodeId, context->getPageToUnpin()->getDbID(), context->getPageToUnpin()->getTypeID(), 
                                      context->getPageToUnpin()->getSetID(), context->getPageToUnpin(), false);
                                  context->setPageToUnpin(output);
                                  context->setNeedUnpin(false);
                              }
                              //std :: cout << "now we allocate a new block" << std :: endl;
                              try {
                                  outputBlock = bundler->loadOutputBlock(batchSize);
                              }
                              catch (NotEnoughSpace &n) {
                                  std :: cout << "current block is full, copy to output page!" << std :: endl;
                                  Record<Vector<Handle<Object>>> * myBytes = getRecord(context->getOutputVec());
                                  memcpy(output->getBytes(), myBytes, myBytes->numBytes());
                                  std :: cout << "we need to unpin the full page" << std :: endl;
                                  proxy->unpinUserPage(nodeId, context->getPageToUnpin()->getDbID(), context->getPageToUnpin()->getTypeID(),
                                      context->getPageToUnpin()->getSetID(), context->getPageToUnpin());
                                  std :: cout << "we need to pin a new page" << std :: endl;
                                  proxy->addUserPage(outputSet->getDatabaseId(), outputSet->getTypeId(), outputSet->getSetId(), output);
                                  std :: cout << "pinned page in output set with id=" << output->getPageID() << std :: endl;
                                  context->setPageToUnpin(output);
                                  makeObjectAllocatorBlock (output->getSize(), true);
                                  outputVec = makeObject<Vector<Handle<Object>>>();
                                  context->setOutputVec(outputVec);
                                  context->setOutputFull(false);
                                  outputBlock = bundler->loadOutputBlock(batchSize); 
                              }
                          }
                          std :: cout << "the input page has been processed" << std :: endl;
                          std :: cout << "run the pipeline on this block" << std :: endl;
                          source->run(context, outputBlock, batchSize);

                          bundler->clearOutputBlock();
                          bundler->clearInputPage();
               
                          std :: cout << "now we unpin the page" << std :: endl;
                          proxy->unpinUserPage(nodeId, page->getDbID(), page->getTypeID(), page->getSetID(), page, false);  
                      }
                  }
                  std :: cout << "outputVec size =" << outputVec->size() << std :: endl;
                  Record<Vector<Handle<Object>>> * myBytes = getRecord(outputVec);
                  PDBPagePtr outputToUnpin = context->getPageToUnpin();
                  if (outputToUnpin == nullptr) {
                       std :: cout << "Error : output page is null in context" << std :: endl;
                       exit(-1);
                  }
                  memcpy(outputToUnpin->getBytes(), myBytes, myBytes->numBytes());
                  //Record <Vector <Handle<Object>>> * myRec = (Record <Vector<Handle<Object>>> *) (context->getPageToUnpin()->getBytes());
                  //Handle<Vector<Handle<Object>>> inputVec = myRec->getRootObject ();
                  //int vecSize = inputVec->size();
                  //std :: cout << "after getRecord: outputVec size =" << vecSize << std :: endl;
                  std :: cout << "unpin the output page" << std :: endl;
                  proxy->unpinUserPage(nodeId, outputToUnpin->getDbID(), outputToUnpin->getTypeID(),
                      outputToUnpin->getSetID(), outputToUnpin, true);
                  std :: cout << "output page is unpinned" << std :: endl;
                  context->setPageToUnpin(nullptr);
                  outputVec = nullptr;
                  std :: cout << "buzz the buzzer" << std :: endl;
                  callerBuzzer->buzz(PDBAlarm :: WorkAllDone, counter);

             }

         );
         worker->execute(myWork, tempBuzzer);
    }

    while (counter < numThreads) {
         tempBuzzer->wait();
    }

    if (server->getFunctionality<HermesExecutionServer>().setCurPageScanner(nullptr) == false) {
        success = false;
        errMsg = "Error: No job is running!";
        std :: cout << errMsg << std :: endl;
        return;
    }


    return;
    

}

}

#endif
