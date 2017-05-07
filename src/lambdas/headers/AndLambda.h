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

#ifndef AND_LAM_H
#define AND_LAM_H

#include <vector>
#include "Lambda.h"
#include "ComputeExecutor.h"
#include "TupleSetMachine.h"
#include "TupleSet.h"
#include "Ptr.h"

namespace pdb {

// only one of these four versions is going to work... used to automatically dereference a Ptr<blah>
// type on either the LHS or RHS of an "and" check
template <class LHS, class RHS>
std :: enable_if_t <std :: is_base_of <PtrBase, LHS> :: value && std :: is_base_of <PtrBase, RHS> :: value, bool> checkAnd (LHS lhs, RHS rhs) {
	return *lhs && *rhs;
} 

template <class LHS, class RHS>
std :: enable_if_t <std :: is_base_of <PtrBase, LHS> :: value && !(std :: is_base_of <PtrBase, RHS> :: value), bool> checkAnd (LHS lhs, RHS rhs) {
	return *lhs && rhs;
} 

template <class LHS, class RHS>
std :: enable_if_t <!(std :: is_base_of <PtrBase, LHS> :: value) && std :: is_base_of <PtrBase, RHS> :: value, bool> checkAnd (LHS lhs, RHS rhs) {
	return lhs && *rhs;
} 

template <class LHS, class RHS>
std :: enable_if_t <!(std :: is_base_of <PtrBase, LHS> :: value) && !(std :: is_base_of <PtrBase, RHS> :: value), bool> checkAnd (LHS lhs, RHS rhs) {
	return lhs && rhs;
} 

template <class LeftType, class RightType> 
class AndLambda : public TypedLambdaObject <bool> {

public:

	LambdaTree <LeftType> lhs;
	LambdaTree <RightType> rhs;

public:

	AndLambda (LambdaTree <LeftType> lhsIn, LambdaTree <RightType> rhsIn) {
		lhs = lhsIn;
		rhs = rhsIn;
                std :: cout << "LHS index is " << lhs.getInputIndex(0) << std :: endl;
                std :: cout << "RHS index is " << rhs.getInputIndex(0) << std :: endl;
                this->setInputIndex (0, lhs.getInputIndex(0));
                this->setInputIndex (1, rhs.getInputIndex(0));
	}

	std :: string getTypeOfLambda () override {
		return std :: string ("&&");
	}

        unsigned int getNumInputs() override {
            return 2;
        }

	int getNumChildren () override {
		return 2;
	}

	GenericLambdaObjectPtr getChild (int which) override {
		if (which == 0)
			return lhs.getPtr ();
		if (which == 1)
			return rhs.getPtr ();
		return nullptr;
	}



	ComputeExecutorPtr getExecutor (TupleSpec &inputSchema, TupleSpec &attsToOperateOn, TupleSpec &attsToIncludeInOutput) override {
	
		// create the output tuple set
		TupleSetPtr output = std :: make_shared <TupleSet> ();

		// create the machine that is going to setup the output tuple set, using the input tuple set
		TupleSetSetupMachinePtr myMachine = std :: make_shared <TupleSetSetupMachine> (inputSchema, attsToIncludeInOutput);

		// these are the input attributes that we will process
		std :: vector <int> inputAtts = myMachine->match (attsToOperateOn);
		int firstAtt = inputAtts[0];
		int secondAtt = inputAtts[1];

		// this is the output attribute
		int outAtt = attsToIncludeInOutput.getAtts ().size ();

		return std :: make_shared <SimpleComputeExecutor> (
			output, 
			[=] (TupleSetPtr input) {

				// set up the output tuple set
				myMachine->setup (input, output);	

				// get the columns to operate on
				std :: vector <LeftType> &leftColumn = input->getColumn <LeftType> (firstAtt);
				std :: vector <RightType> &rightColumn = input->getColumn <RightType> (secondAtt);

				// create the output attribute, if needed
				if (!output->hasColumn (outAtt)) { 
					std :: vector <bool> *outColumn = new std :: vector <bool>;
					output->addColumn (outAtt, outColumn, true); 
				}

				// get the output column
				std :: vector <bool> &outColumn = output->getColumn <bool> (outAtt);

				// loop down the columns, setting the output
				int numTuples = leftColumn.size ();
				outColumn.resize (numTuples); 
				for (int i = 0; i < numTuples; i++) {
					outColumn [i] = checkAnd (leftColumn[i], rightColumn[i]);
				}
				return output;
			}
		);
		
	}

        ComputeExecutorPtr getRightOneHasher (TupleSpec &inputSchema, TupleSpec &attsToOperateOn, TupleSpec &attsToIncludeInOutput) override {

                // create the output tuple set
                TupleSetPtr output = std :: make_shared <TupleSet> ();

                // create the machine that is going to setup the output tuple set, using the input tuple set
                TupleSetSetupMachinePtr myMachine = std :: make_shared <TupleSetSetupMachine> (inputSchema, attsToIncludeInOutput);

                // these are the input attributes that we will process
                std :: vector <int> inputAtts = myMachine->match (attsToOperateOn);
                int secondAtt = inputAtts[0];

                // this is the output attribute
                int outAtt = attsToIncludeInOutput.getAtts ().size ();

                return std :: make_shared <SimpleComputeExecutor> (
                        output,
                        [=] (TupleSetPtr input) {

                                // set up the output tuple set
                                myMachine->setup (input, output);

                                // get the columns to operate on
                                std :: vector <RightType> &rightColumn = input->getColumn <RightType> (secondAtt);

                                // create the output attribute, if needed
                                if (!output->hasColumn (outAtt)) {
                                        std :: vector <size_t> *outColumn = new std :: vector <size_t>;
                                        output->addColumn (outAtt, outColumn, true);
                                }

                                // get the output column
                                std :: vector <size_t> &outColumn = output->getColumn <size_t> (outAtt);

                                // loop down the columns, setting the output
                                int numTuples = rightColumn.size ();
                                outColumn.resize (numTuples);
                                for (int i = 0; i < numTuples; i++) {
                                        outColumn [i] = 1;
                                }
                                return output;
                        }
                );
        }




        ComputeExecutorPtr getLeftOneHasher (TupleSpec &inputSchema, TupleSpec &attsToOperateOn, TupleSpec &attsToIncludeInOutput) override {

                // create the output tuple set
                TupleSetPtr output = std :: make_shared <TupleSet> ();

                // create the machine that is going to setup the output tuple set, using the input tuple set
                TupleSetSetupMachinePtr myMachine = std :: make_shared <TupleSetSetupMachine> (inputSchema, attsToIncludeInOutput);

                // these are the input attributes that we will process
                std :: vector <int> inputAtts = myMachine->match (attsToOperateOn);
                int firstAtt = inputAtts[0];

                // this is the output attribute
                int outAtt = attsToIncludeInOutput.getAtts ().size ();

                return std :: make_shared <SimpleComputeExecutor> (
                        output,
                        [=] (TupleSetPtr input) {

                                // set up the output tuple set
                                myMachine->setup (input, output);

                                // get the columns to operate on
                                std :: vector <LeftType> &leftColumn = input->getColumn <LeftType> (firstAtt);

                                // create the output attribute, if needed
                                if (!output->hasColumn (outAtt)) {
                                        std :: vector <size_t> *outColumn = new std :: vector <size_t>;
                                        output->addColumn (outAtt, outColumn, true);
                                }

                                // get the output column
                                std :: vector <size_t> &outColumn = output->getColumn <size_t> (outAtt);

                                // loop down the columns, setting the output
                                int numTuples = leftColumn.size ();
                                outColumn.resize (numTuples);
                                for (int i = 0; i < numTuples; i++) {
                                        outColumn [i] = 1;
                                }
                                return output;
                        }
                );
        }




};

}

#endif
