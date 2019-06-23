//
// Created by vicram on 6/23/19.
//

#ifndef PDB_ABSTRACTCONVERTER_H
#define PDB_ABSTRACTCONVERTER_H

/**
 * This is an interface for a class that has a single method, convert. It
 * is used by AggregateComp.
 * @tparam In Input type.
 * @tparam Out Output type.
 */
template<class In, class Out>
class AbstractConverter {
 public:
  /**
   * This method converts an object of type In to an object of type Out.
   * There is no return value; the output should be stored at location out.
   * @param in The object to be converted, passed by reference.
   * @param out Pointer to the location where the output should be stored.
   */
  virtual void convert(In& in, Out* out) = 0;
};

#endif //PDB_ABSTRACTCONVERTER_H
