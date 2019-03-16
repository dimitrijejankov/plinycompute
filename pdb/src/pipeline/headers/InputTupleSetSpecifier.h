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
#ifndef INPUT_TUPLESET_SPECIFIER_HEADER
#define INPUT_TUPLESET_SPECIFIER_HEADER

#include <iostream>
#include <string>
#include <vector>

namespace pdb {

class InputTupleSetSpecifier {

public:

//default constructor
InputTupleSetSpecifier ();


//constructor
InputTupleSetSpecifier (std :: string tupleSetName, std :: vector < std :: string > columnNamesToKeep, std :: vector < std :: string > columnNamesToApply);

//destructor
~InputTupleSetSpecifier ();

//return tuple set name
std :: string & getTupleSetName ();

//return column names to keep in the output

std :: vector  < std :: string > & getColumnNamesToKeep ();

//return column names to apply a lambda

std :: vector  < std :: string > & getColumnNamesToApply ();

void print();

void clear();

private:

//name of the the tuple set
std :: string tupleSetName;

//column names in the tuple set to keep in the output
std :: vector < std :: string > columnNamesToKeep;

//column names in the tuple set to apply
std :: vector < std :: string > columnNamesToApply;

};

}

#endif
