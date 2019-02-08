
#ifndef FILE_WRITER_H
#define FILE_WRITER_H

#include <map>
#include <memory>
#include <string>
#include <vector>

using namespace std;

namespace pdb {

// this encapsulates a simple little key-value store
class PDBBufferManagerFileWriter {

 public:

  // these functions search the file for the specified key, and
  // they return the value stored in the file using the specified
  // type.  A true is returned on success.  A false is returned on
  // error (the key is not found, or the value cannot be cast to the
  // specified type)
  bool getUnsignedLong(string key, uint64_t &value);
  bool getLong(string key, int64_t &value);
  bool getString (string key, string &value);
  bool getStringList (string key, vector <string> &value);

  // these functions add a new (key, value) pair into the file.
  // If the key is already in the file, then its value is replaced.
  void putString (string key, string value);
  void putStringList (string key, vector <string> value);
  void putUnsignedLong(string key, uint64_t value);
  void putLong(string key, int64_t value);

  // creates an instance of the file.  If the specified file does
  // not exist, it is created.  Otherwise, the existing file is
  // opened.
  explicit PDBBufferManagerFileWriter (string fName);

  // closes the file, and saves the contents.
  ~PDBBufferManagerFileWriter () = default;

  // saves any updates to the file
  void save ();

 private:

  // the name of the file to read/write
  string fName;

  // the map that stores the writer's contents
  map <string, string> myData;
};

}



#endif
