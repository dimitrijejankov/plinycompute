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
#include "QueriesTestsRunner.h"

#include "BuildIrTests.h"
#include "DotBuilderTests.h"
#include "ProjectionIrTests.h"
#include "SelectionIrTests.h"

namespace pdb_tests {
void runQueryIrTests(UnitTest& qunit) {
    testDotBuilderSelection(qunit);
    testBuildIrSelection1(qunit);
    testBuildIrSelection2(qunit);
    testBuildIrSelection3(qunit);
    testProjectionIrExecute(qunit);
    testSelectionIrExecute(qunit);
}
}
