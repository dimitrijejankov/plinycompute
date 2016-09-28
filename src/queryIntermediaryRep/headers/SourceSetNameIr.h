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
#ifndef PDB_QUERYINTERMEDIARYREP_SOURCESETNAMEIR_H
#define PDB_QUERYINTERMEDIARYREP_SOURCESETNAMEIR_H

#include "Handle.h"
#include "PDBString.h"
#include "SetExpressionIr.h"

using pdb::Handle;
using pdb::String;

namespace pdb_detail
{
    /**
     * A PDB set identified by database name and set name used as original input to a query.
     */
    class SourceSetNameIr : public SetExpressionIr
    {

    public:

        SourceSetNameIr();

        /**
         * Creates a set name from the given database name and set name.
         *
         * @param databaseName the database name.
         * @param setName the set name.
         * @return the set name.
         */
        SourceSetNameIr(string databaseName, string setName);

        // contract from super
        void execute(SetExpressionIrAlgo &algo) override;

        /**
         * @return the name of the database containing the set.
         */
        string getDatabaseName();

        /**
         * @return the name of the set.
         */
        string getName();



    private:

        /**
         * The name of the database that contains the set.
         */
        string _databaseName;

        /**
         * The name of the set in the database.
         */
        string _setName;
    };
}

#endif //PDB_QUERYINTERMEDIARYREP_SETNAMEIR_H
