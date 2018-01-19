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

#include <iostream>

#include "LogicalPlanTestsRunner.h"
#include "InterfaceFunctions.h"
#include "QueryItermediaryRepTestsRunner.h"
#include "QueriesTestsRunner.h"
#include "TcapBuilderTestsRunner.h"
#include "TcapTestsRunner.h"
#include "TcapParsersTestsRunner.h"
#include "TcapIrTestsRunner.h"
#include "qunit.h"


using QUnit::UnitTest;

using pdb::makeObjectAllocatorBlock;

using pdb_tests::runQueriesTests;
using pdb_tests::runQueryIrTests;
using pdb_tests::runTcapTests;
using pdb_tests::runTcapParserTests;
using pdb_tests::runBuildTcapIrTests;
using pdb_tests::runLogicalPlanTests;
using pdb_tests::runBuildTcapBuilderTests;

int main() {
    makeObjectAllocatorBlock(1024 * 10, true);

    UnitTest qunit(std::cerr, QUnit::normal);

    runQueriesTests(qunit);
    runQueryIrTests(qunit);
    runTcapTests(qunit);
    runTcapParserTests(qunit);
    runBuildTcapIrTests(qunit);
    runBuildTcapBuilderTests(qunit);
    runLogicalPlanTests(qunit);

    return qunit.errors();
}