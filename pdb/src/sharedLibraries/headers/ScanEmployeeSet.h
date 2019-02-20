//
// Created by dimitrije on 2/20/19.
//

#ifndef PDB_SCANEMPLOYEESET_H
#define PDB_SCANEMPLOYEESET_H

#include <Employee.h>
#include <ScanSet.h>
#include <LambdaCreationFunctions.h>
#include <VectorTupleSetIterator.h>

class ScanEmployeeSet : public pdb::ScanSet<pdb::Employee> {

 public:

  ENABLE_DEEP_COPY

  ScanEmployeeSet() = default;

  // eventually, this method should be moved into a class that works with the system to
  // iterate through pages that are pulled from disk/RAM by the system... a programmer
  // should not provide this particular method
  pdb::ComputeSourcePtr getComputeSource(TupleSpec &schema, pdb::ComputePlan &plan) override {

    return std::make_shared<pdb::VectorTupleSetIterator>(

        // constructs a list of data objects to iterate through
        []() -> void * {

          // this implementation only serves six pages
          static int numPages = 0;
          if (numPages == 6)
            return nullptr;

          // create a page, loading it with random data
          void *myPage = malloc(1024 * 1024);
          {
            const pdb::UseTemporaryAllocationBlock tempBlock{myPage, 1024 * 1024};

            // write a bunch of supervisors to it
            pdb::Handle<pdb::Vector<pdb::Handle<pdb::Employee>>> employees = pdb::makeObject<pdb::Vector<pdb::Handle<pdb::Employee>>>();

            // this will build up the department
            char first = 'A', second = 'B';
            char myString[3];
            myString[2] = 0;

            try {
              for (int i = 0; true; i++) {

                myString[0] = first;
                myString[1] = second;

                // this will allow us to cycle through "AA", "AB", "AC", "BA", ...
                first++;
                if (first == 'D') {
                  first = 'A';
                  second++;
                  if (second == 'D')
                    second = 'A';
                }

                if(i % 2 == 0) {

                  pdb::Handle<pdb::Employee> temp = pdb::makeObject<pdb::Employee>("Steve Stevens", 20 + ((i) % 29), std::string(myString), i * 3.54);
                  employees->push_back(temp);
                }
                else {
                  pdb::Handle<pdb::Employee> temp = pdb::makeObject<pdb::Employee>("Ninja Turtles", 20 + ((i) % 29), std::string(myString), i * 3.54);
                  employees->push_back(temp);
                }
              }
            } catch (pdb::NotEnoughSpace &e) {

              getRecord (employees);
            }
          }
          numPages++;
          return myPage;
        },

        // frees the list of data objects that have been iterated
        [](void *freeMe) -> void {
          free(freeMe);
        },

        // and this is the chunk size, or number of items to put into each tuple set
        24);
  }
};

#endif //PDB_SCANEMPLOYEESET_H
