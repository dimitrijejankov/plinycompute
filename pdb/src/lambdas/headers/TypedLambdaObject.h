//
// Created by dimitrije on 2/19/19.
//

#ifndef PDB_TYPEDLAMBDAOBJECT_H
#define PDB_TYPEDLAMBDAOBJECT_H

#include "LambdaObject.h"

namespace pdb {

// this is the lamda type... queries are written by supplying code that
// creates these objects
template<typename Out>
class TypedLambdaObject : public LambdaObject {

public:

  ~TypedLambdaObject() override = default;

  std::string getOutputType() override {
    return getTypeName<Out>();
  }
};

}
#endif //PDB_TYPEDLAMBDAOBJECT_H
