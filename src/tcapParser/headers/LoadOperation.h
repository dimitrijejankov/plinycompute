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
#ifndef PDB_TCAPPARSER_LOADOPERATION_H
#define PDB_TCAPPARSER_LOADOPERATION_H

#include <memory>
#include <string>

#include "TableExpression.h"

using std::shared_ptr;
using std::string;

namespace pdb_detail
{
    class LoadOperation : public TableExpression
    {
    public:

        LoadOperation(shared_ptr<string> source) ;

        LoadOperation(string source);

        shared_ptr<string> source;

        void execute(function<void(LoadOperation&)> forLoad, function<void(ApplyOperation&)>,
                     function<void(FilterOperation&)>,function<void(HoistOperation&)> forHoist,
                     function<void(BinaryOperation&)> forBinaryOp) override;

    };
}

#endif //PDB_TCAPPARSER_LOADOPERATION_H
