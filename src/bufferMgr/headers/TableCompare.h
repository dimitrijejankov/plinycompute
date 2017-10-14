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


/****************************************************
** COPYRIGHT 2016, Chris Jermaine, Rice University **
**                                                 **
** The MyDB Database System, COMP 530              **
** Note that this file contains SOLUTION CODE for  **
** A1.  You should not be looking at this file     **
** unless you have completed A1!                   **
****************************************************/

#ifndef TABLE_COMP_H
#define TABLE_COMP_H

#include "MyDB_Table.h"

// so that pages can be put into a map
struct TableCompare {

public:
    bool operator()(const MyDB_TablePtr lhs, const MyDB_TablePtr rhs) const {

        // deal with the null case
        if (lhs == nullptr && rhs != nullptr) {
            return true;
        } else if (rhs == nullptr) {
            return false;
        }

        // otherwise, just compare the strings
        return lhs->getName() < rhs->getName();
    }
};

#endif
