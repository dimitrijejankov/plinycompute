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
#include "TableAssignment.h"

namespace pdb_detail
{
    TableAssignment::TableAssignment(TcapIdentifier tableName, shared_ptr<vector<TcapIdentifier>> columnNames,  shared_ptr<TableExpression> value)
            : tableName(tableName), columnNames(columnNames), value(value)
    {
    }

    void TableAssignment::match(function<void(TableAssignment&)> forTableAssignment, function<void(StoreOperation&)>)
    {
        forTableAssignment(*this);
    }

}