#include <fstream>
#include <vector>
#include <PDBClient.h>
#include <GenericWork.h>
#include "sharedLibraries/headers/FFMatrixBlock.h"
#include "sharedLibraries/headers/FFMatrixScanner.h"
#include "sharedLibraries/headers/FFMatrixWriter.h"
#include "sharedLibraries/headers/FFInputLayerJoin.h"
#include "sharedLibraries/headers/FFAggMatrix.h"
#include "sharedLibraries/headers/FFHiddenLayerJoin.h"
#include "sharedLibraries/headers/FFJoinBackTransposeMult.h"
#include "sharedLibraries/headers/FFSelectionGradient2.h"
#include "sharedLibraries/headers/FFGradientJoin.h"
#include "sharedLibraries/headers/FFUpdateJoin.h"
#include "sharedLibraries/headers/FFJoinBackMultTranspose.h"

using namespace pdb;

std::vector<std::vector<std::pair<int32_t, int32_t>>> labels;

//
std::vector<std::pair<int32_t, int32_t>> labels_meta;
std::vector<std::pair<int32_t, int32_t>> labels_data;

std::vector<std::vector<std::pair<int32_t, float>>> features;

int32_t total_points;

int32_t num_batch;
int32_t batch_block;

int32_t num_features;
int32_t features_block;

int32_t num_labels;
int32_t labels_block;

int32_t embedding_size;
int32_t embedding_block;

bool read_label(std::ifstream &is, char *buffer, int32_t batch_id) {

  int idx = 0;
  do {

    is.read(&buffer[idx], 1);
    if (buffer[idx] == ' ' || buffer[idx] == ',') {

      // add the end
      buffer[idx + 1] = '\0';

      // read the label
      char *tmp;
      auto label_id = strtol(buffer, &tmp, 10);

      // get the row id and col id
      auto row_id = batch_id / batch_block;
      auto col_id = label_id / labels_block;

      // store the labels
      labels[row_id * (num_labels / labels_block) + col_id].emplace_back(std::make_pair(batch_id % batch_block,
                                                                         label_id % labels_block));

      // does not have other labels
      return buffer[idx] == ',';
    }

    // increment the index
    idx++;

  } while (true);
}

void read_feature_idx(std::ifstream &is, char *buffer, int32_t pos) {

  int idx = 0;
  do {

    is.read(&buffer[idx], 1);
    if (buffer[idx] == ':') {

      // add the end
      buffer[idx + 1] = '\0';

      // read the label
      char *tmp;
      features[pos].emplace_back();
      features[pos].back().first = strtol(buffer, &tmp, 10);

      // does not have other labels
      return;
    }

    // increment the index
    idx++;

  } while (true);
}

bool read_feature_value(std::ifstream &is, char *buffer, int32_t pos) {

  int idx = 0;
  do {

    is.read(&buffer[idx], 1);
    if (buffer[idx] == ' ' || buffer[idx] == '\n') {

      // add the end
      buffer[idx + 1] = '\0';

      // read the label
      std::string::size_type tmp;
      features[pos].back().second = stof(buffer, &tmp);

      // does not have other labels
      return buffer[idx] == ' ';
    }

    // increment the index
    idx++;

  } while (true);
}

void init_features(int32_t row_id, int32_t col_id, float *values) {

  auto start_r = row_id * batch_block;
  auto end_r = (row_id + 1) * batch_block;

  for(int r = start_r; r < end_r; ++r) {

    // get the features row
    auto &features_row = features[r];

    // get the start and end column
    auto start_c = col_id * features_block;
    auto end_c = (col_id + 1) * features_block;

    for(const auto &f : features_row) {
      if(f.first >= start_c && f.first < end_c) {
        values[(r % batch_block) * features_block + (f.first % features_block)] = f.second;
      }
    }
  }

}

void load_input_data(pdb::PDBClient &pdbClient) {

  /// 1. Load the data from the file

  // open the input file
  std::ifstream is("/home/ubuntu/disk/data/input.txt");
  //std::ifstream is("./applications/FFTest/eurlex_train.txt");

  // load the data stats
  is >> total_points;
  is >> num_features;
  is >> num_labels;

  // round features so we can pad them
  if((num_features % features_block) != 0) {
    num_features += features_block - (num_features % features_block);
  }

  // round labels so we can pad them
  if((num_labels % labels_block) != 0) {
    num_labels += labels_block - (num_labels % labels_block);
  }

  // check that we have enough data points
  if (total_points < num_batch) {
    throw runtime_error("Not enough data points to form a batch " + std::to_string(total_points));
  }

  // init the data
  labels.resize((num_batch / batch_block) * (num_labels / labels_block));
  features.resize(num_batch);

  // load a single batch into memory
  char buffer[1024];
  for (int i = 0; i < num_batch; ++i) {

    // read the labels
    while (read_label(is, buffer, i)) {}

    do {

      // try to read the feature index
      read_feature_idx(is, buffer, i);

      // try to read the feature value
      if (!read_feature_value(is, buffer, i)) {
        break;
      }

    } while (true);
  }

  /// 2. Send the input batch block to the sever

  // figure out how many blocks we need to generate
  int32_t batch_s = num_batch / batch_block;
  int32_t batch_f = num_features / features_block;

  // figure out all the blocks we need to send
  std::vector<std::pair<int32_t, int32_t>> tuples_to_send(batch_s * batch_f);
  for (int i = 0; i < batch_s; ++i) {
    for (int j = 0; j < batch_f; ++j) {
      tuples_to_send[i * batch_f + j] = {i, j};
    }
  }

  size_t idx = 0;
  while (idx != tuples_to_send.size()) {

    // use temporary allocation block
    const pdb::UseTemporaryAllocationBlock tempBlock{64 * 1024 * 1024};

    // put the chunks here
    Handle<Vector<Handle<ff::FFMatrixBlock>>> vec = pdb::makeObject<Vector<Handle<ff::FFMatrixBlock>>>();

    try {

      // put stuff into the vector
      for (; idx < tuples_to_send.size();) {

        // allocate a matrix
        Handle<ff::FFMatrixBlock> myInt = makeObject<ff::FFMatrixBlock>(tuples_to_send[idx].first,
                                                                        tuples_to_send[idx].second,
                                                                        batch_block,
                                                                        features_block);

        // init the values
        float *vals = myInt->data->data->c_ptr();
        bzero(myInt->data->data->c_ptr(), batch_block * features_block * sizeof(float));

        init_features(tuples_to_send[idx].first, tuples_to_send[idx].second, myInt->data->data->c_ptr());

        // we add the matrix to the block
        vec->push_back(myInt);

        // go to the next one
        ++idx;

        if (vec->size() == 50) {
          break;
        }
      }
    }
    catch (pdb::NotEnoughSpace &n) {}

    // init the records
    getRecord(vec);

    // send the data a bunch of times
    pdbClient.sendData<ff::FFMatrixBlock>("ff", "input_batch", vec);

    // log that we stored stuff
    std::cout << "stored in input batch " << vec->size() << " !\n";
  }

  /// 3. Send the label batch block to the sever

  int32_t batch_l = num_labels / labels_block;

  labels_meta.resize(batch_s * batch_l);

  // figure out all the blocks we need to send
  idx = 0;
  for (int i = 0; i < batch_s; ++i) {
    for (int j = 0; j < batch_l; ++j) {

      //                                pos,                num
      labels_meta[i * batch_l + j] = {labels_data.size(), labels[i * batch_l + j].size()};

      // insert the labels data
      labels_data.insert(labels_data.end(), labels[i * batch_l + j].begin(), labels[i * batch_l + j].end());
    }
  }
}

void generate_input_data(pdb::PDBClient &pdbClient) {

  /// 1. Load the data from the file

  // round features so we can pad them
  if((num_features % features_block) != 0) {
    num_features += features_block - (num_features % features_block);
  }

  // round labels so we can pad them
  if((num_labels % labels_block) != 0) {
    num_labels += labels_block - (num_labels % labels_block);
  }

  /// 2. Send the input batch block to the sever

  // figure out how many blocks we need to generate
  int32_t batch_s = num_batch / batch_block;
  int32_t batch_f = num_features / features_block;

  // figure out all the blocks we need to send
  std::vector<std::pair<int32_t, int32_t>> tuples_to_send(batch_s * batch_f);
  for (int i = 0; i < batch_s; ++i) {
    for (int j = 0; j < batch_f; ++j) {
      tuples_to_send[i * batch_f + j] = {i, j};
    }
  }

  size_t idx = 0;
  while (idx != tuples_to_send.size()) {

    // use temporary allocation block
    const pdb::UseTemporaryAllocationBlock tempBlock{64 * 1024 * 1024};

    // put the chunks here
    Handle<Vector<Handle<ff::FFMatrixBlock>>> vec = pdb::makeObject<Vector<Handle<ff::FFMatrixBlock>>>();

    try {

      // put stuff into the vector
      for (; idx < tuples_to_send.size();) {

        // allocate a matrix
        Handle<ff::FFMatrixBlock> myInt = makeObject<ff::FFMatrixBlock>(tuples_to_send[idx].first,
                                                                        tuples_to_send[idx].second,
                                                                        batch_block,
                                                                        features_block);

        // init the values
        float *vals = myInt->data->data->c_ptr();
        for(int i = 0; i < batch_block * features_block; ++i) {
          vals[i] = drand48();
        }

        // we add the matrix to the block
        vec->push_back(myInt);

        // go to the next one
        ++idx;

        if (vec->size() == 50) {
          break;
        }
      }
    }
    catch (pdb::NotEnoughSpace &n) {}

    // init the records
    getRecord(vec);

    // send the data a bunch of times
    pdbClient.sendData<ff::FFMatrixBlock>("ff", "input_batch", vec);

    // log that we stored stuff
    std::cout << "stored in input batch " << vec->size() << " !\n";
  }

  /// 3. Send the label batch block to the sever

  int32_t batch_l = num_labels / labels_block;

  // figure out all the blocks we need to send
  labels_meta.resize(batch_s * batch_l);
  for (int i = 0; i < batch_s; ++i) {
    for (int j = 0; j < batch_l; ++j) {

      // make a count
      int32_t cnt = rand() % 100;

      // fill up random meta data     pos,                num
      labels_meta[i * batch_l + j] = {labels_data.size(), cnt};

      // fill up some random data
      for(int32_t t = 0; t < cnt; ++t) {
        labels_data.emplace_back(rand() % num_batch, rand() % labels_block);
      }
    }
  }
}

auto init_weights(pdb::PDBClient &pdbClient) {

  /// 1. Init the word embedding a matrix of size (num_features x embedding_block)

  auto block_f = num_features / features_block;
  auto block_e = embedding_size / embedding_block;

  // the page size
  const int32_t page_size = 64;

  // all the block we need to send
  std::vector<std::pair<int32_t, int32_t>> tuples_to_send(block_f * block_e);
  for (int i = 0; i < block_f; ++i) {
    for (int j = 0; j < block_e; ++j) {
      tuples_to_send[i * block_e + j] = {i, j};
    }
  }

  size_t idx = 0;
  while (idx != tuples_to_send.size()) {

    // use temporary allocation block
    const pdb::UseTemporaryAllocationBlock tempBlock{page_size * 1024 * 1024};

    // put the chunks here
    Handle<Vector<Handle<ff::FFMatrixBlock>>> data = pdb::makeObject<Vector<Handle<ff::FFMatrixBlock>>>();

    try {

      // put stuff into the vector
      for (; idx < tuples_to_send.size();) {

        // allocate a matrix
        Handle<ff::FFMatrixBlock> myInt = makeObject<ff::FFMatrixBlock>(tuples_to_send[idx].first,
                                                                        tuples_to_send[idx].second,
                                                                        features_block,
                                                                        embedding_block);

        // init the values
        float *vals = myInt->data->data->c_ptr();
        for (int v = 0; v < features_block * embedding_block; ++v) {
          vals[v] = (float) drand48() * 0.1f;
        }

        // check if we need to init the bias here if so do it...
        if(tuples_to_send[idx].first == (block_f - 1)) {

          // allocate the bias if necessary
          myInt->data->bias = makeObject<Vector<float>>(embedding_block, embedding_block);

          // init the bias
          for(int i = 0; i < embedding_block; ++i) {
            myInt->data->bias->c_ptr()[i] = (float) drand48() * 0.1f;
          }
        }

        // we add the matrix to the block
        data->push_back(myInt);

        // go to the next one
        ++idx;

        if (data->size() == 50) {
          break;
        }
      }
    }
    catch (pdb::NotEnoughSpace &n) {}

    // init the records
    getRecord(data);

    // send the data a bunch of times
    pdbClient.sendData<ff::FFMatrixBlock>("ff", "w1", data);

    // log that we stored stuff
    std::cout << "stored in embedding " << data->size() << " !\n";
  }

  /// 2. Init the dense layer (embedding_block x block_l)

  // how many blocks we split the labels
  auto block_l = num_labels / labels_block;

  tuples_to_send.resize(block_e * block_l);
  for (int i = 0; i < block_e; ++i) {
    for (int j = 0; j < block_l; ++j) {
      tuples_to_send[i * block_l + j] = {i, j};
    }
  }

  idx = 0;
  while (idx != tuples_to_send.size()) {

    // use temporary allocation block
    const pdb::UseTemporaryAllocationBlock tempBlock{page_size * 1024 * 1024};

    // put the chunks here
    Handle<Vector<Handle<ff::FFMatrixBlock>>> data = pdb::makeObject<Vector<Handle<ff::FFMatrixBlock>>>();

    try {

      // put stuff into the vector
      for (; idx < tuples_to_send.size();) {

        // allocate a matrix
        Handle<ff::FFMatrixBlock> myInt = makeObject<ff::FFMatrixBlock>(tuples_to_send[idx].first,
                                                                        tuples_to_send[idx].second,
                                                                        embedding_block,
                                                                        labels_block);

        // init the values
        float *vals = myInt->data->data->c_ptr();
        for (int v = 0; v < embedding_block * labels_block; ++v) {
          vals[v] = (float) drand48() * 0.1f;
        }

        if(tuples_to_send[idx].first == (block_e - 1)) {

          // init the bias if necessary
          myInt->data->bias = makeObject<Vector<float>>(labels_block, labels_block);

                    // init the bias
          for(int i = 0; i < labels_block; ++i) {
            myInt->data->bias->c_ptr()[i] = (float) drand48() * 0.1f;
          }
        }

        // we add the matrix to the block
        data->push_back(myInt);

        // go to the next one
        ++idx;

        if (data->size() == 50) {
          break;
        }
      }
    }
    catch (pdb::NotEnoughSpace &n) {}

    // init the records
    getRecord(data);

    // send the data a bunch of times
    pdbClient.sendData<ff::FFMatrixBlock>("ff", "w2", data);

    // log that we stored stuff
    std::cout << "stored in dense " << data->size() << " !\n";
  }
}

int main(int argc, char *argv[]) {

  // make a client
  pdb::PDBClient pdbClient(8108, "localhost");

  // now, register a type for user data
  pdbClient.registerType("libraries/libFFHiddenLayerJoin.so");
  pdbClient.registerType("libraries/libFFAggMatrix.so");
  pdbClient.registerType("libraries/libFFGradientJoin.so");
  pdbClient.registerType("libraries/libFFHiddenLayerJoin.so");
  pdbClient.registerType("libraries/libFFInputLayerJoin.so");
  pdbClient.registerType("libraries/libFFJoinBackMultTranspose.so");
  pdbClient.registerType("libraries/libFFJoinBackTransposeMult.so");
  pdbClient.registerType("libraries/libFFMatrixBlock.so");
  pdbClient.registerType("libraries/libFFMatrixData.so");
  pdbClient.registerType("libraries/libFFMatrixMeta.so");
  pdbClient.registerType("libraries/libFFMatrixScanner.so");
  pdbClient.registerType("libraries/libFFMatrixWriter.so");
  pdbClient.registerType("libraries/libFFSelectionGradient2.so");
  pdbClient.registerType("libraries/libFFUpdateJoin.so");


  // now, create a new database
  pdbClient.createDatabase("ff");

  // now, create the input and output sets
  pdbClient.createSet<ff::FFMatrixBlock>("ff", "input_batch");
  pdbClient.createSet<ff::FFMatrixBlock>("ff", "output_labels");
  pdbClient.createSet<ff::FFMatrixBlock>("ff", "w1");
  pdbClient.createSet<ff::FFMatrixBlock>("ff", "w2");

  pdbClient.createSet<ff::FFMatrixBlock>("ff", "activation_1"); // OK
  pdbClient.createSet<ff::FFMatrixBlock>("ff", "activation_2"); // OK
  pdbClient.createSet<ff::FFMatrixBlock>("ff", "gradient_2"); // OK
  pdbClient.createSet<ff::FFMatrixBlock>("ff", "d_w2"); // OK
  pdbClient.createSet<ff::FFMatrixBlock>("ff", "gradient_1_tmp"); // OK
  pdbClient.createSet<ff::FFMatrixBlock>("ff", "gradient_1"); // OK
  pdbClient.createSet<ff::FFMatrixBlock>("ff", "d_w1"); // OK
  pdbClient.createSet<ff::FFMatrixBlock>("ff", "w1_updated"); // OK
  pdbClient.createSet<ff::FFMatrixBlock>("ff", "w2_updated"); // OK

  // should we genrate the data
  bool shouldGenerate;
  std::cout << "Should generate data : \n";
  std::cin >> shouldGenerate;

  if(shouldGenerate) {

    std::cout << "num_batch : \n";
    std::cin >> num_batch;

    std::cout << "batch_block : \n";
    std::cin >> batch_block;

    std::cout << "num_features : \n";
    std::cin >> num_features;

    std::cout << "features_block : \n";
    std::cin >> features_block;

    std::cout << "num_labels : \n";
    std::cin >> num_labels;

    std::cout << "labels_block : \n";
    std::cin >> labels_block;

    std::cout << "embedding_size : \n";
    std::cin >> embedding_size;

    std::cout << "embedding_block : \n";
    std::cin >> embedding_block;

    // generate the input data
    generate_input_data(pdbClient);
  }
  else {

    std::cout << "num_batch : \n";
    std::cin >> num_batch;

    std::cout << "batch_block : \n";
    std::cin >> batch_block;

    std::cout << "features_block : \n";
    std::cin >> features_block;

    std::cout << "labels_block : \n";
    std::cin >> labels_block;

    std::cout << "embedding_size : \n";
    std::cin >> embedding_size;

    std::cout << "embedding_block : \n";
    std::cin >> embedding_block;

    // load the input data
    load_input_data(pdbClient);
  }
  // initialize the weights
  init_weights(pdbClient);

  std::chrono::steady_clock::time_point planner_begin = std::chrono::steady_clock::now();

  // do the activation of the first layer
  std::chrono::steady_clock::time_point stage_begin = std::chrono::steady_clock::now();
  {
    const UseTemporaryAllocationBlock tempBlock{1024 * 1024 * 128};

    // make the computation
    Handle<Computation> readA = makeObject<ff::FFMatrixScanner>("ff", "input_batch");
    Handle<Computation> readB = makeObject<ff::FFMatrixScanner>("ff", "w1");

    // make the join
    Handle<Computation> join = makeObject<ff::FFInputLayerJoin>();
    join->setInput(0, readA);
    join->setInput(1, readB);

    // make the aggregation
    Handle<Computation> myAggregation = makeObject<ff::FFAggMatrix>();
    myAggregation->setInput(join);

    // make the writer
    Handle<Computation> myWriter = makeObject<ff::FFMatrixWriter>("ff", "activation_1");
    myWriter->setInput(myAggregation);

    // run the computation
    bool success = pdbClient.executeComputations({myWriter});
  }
  std::chrono::steady_clock::time_point stage_end = std::chrono::steady_clock::now();
  std::cout << "Run activation_1 for " << std::chrono::duration_cast<std::chrono::nanoseconds>(stage_end - stage_begin).count() << "[ns]" << '\n';

  // do the activation of the second layer
  stage_begin = std::chrono::steady_clock::now();
  {
    // do the activation of the first layer
    const UseTemporaryAllocationBlock tempBlock{1024 * 1024 * 128};

    // make the computation
    Handle<Computation> readA = makeObject<ff::FFMatrixScanner>("ff", "activation_1");
    Handle<Computation> readB = makeObject<ff::FFMatrixScanner>("ff", "w2");

    // make the join
    Handle<Computation> join = makeObject<ff::FFHiddenLayerJoin>();
    join->setInput(0, readA);
    join->setInput(1, readB);

    // make the aggregation
    Handle<Computation> myAggregation = makeObject<ff::FFAggMatrix>();
    myAggregation->setInput(join);

    // make the writer
    Handle<Computation> myWriter = makeObject<ff::FFMatrixWriter>("ff", "activation_2");
    myWriter->setInput(myAggregation);

    // run the computation
    bool success = pdbClient.executeComputations({myWriter});
  }
  stage_end = std::chrono::steady_clock::now();
  std::cout << "Run activation_2 for " << std::chrono::duration_cast<std::chrono::nanoseconds>(stage_end - stage_begin).count() << "[ns]" << '\n';

  // calculate the gradient_2
  stage_begin = std::chrono::steady_clock::now();
  {
    // do the activation of the first layer
    const UseTemporaryAllocationBlock tempBlock{1024 * 1024 * 128};

    // make the computation
    Handle<Computation> lhs = makeObject<ff::FFMatrixScanner>("ff", "activation_2");

    // make the join
    Handle<Computation> selection = makeObject<ff::FFSelectionGradient2>(num_batch / batch_block,
                                                                         num_labels / labels_block,
                                                                         labels_meta,
                                                                         labels_data);
    selection->setInput(lhs);

    // make the writer
    Handle<Computation> writer = makeObject<ff::FFMatrixWriter>("ff", "gradient_2");
    writer->setInput(selection);

    // run the computation
    bool success = pdbClient.executeComputations({writer});
  }
  stage_end = std::chrono::steady_clock::now();
  std::cout << "Run gradient_2 for " << std::chrono::duration_cast<std::chrono::nanoseconds>(stage_end - stage_begin).count() << "[ns]" << '\n';

  // remove activation_2 it is not needed anymore...
  pdbClient.removeSet("ff", "activation_2");

  // calculate d_w2
  stage_begin = std::chrono::steady_clock::now();
  {
    // do the activation of the first layer
    const UseTemporaryAllocationBlock tempBlock{1024 * 1024 * 128};

    // make the computation
    Handle<Computation> readA = makeObject<ff::FFMatrixScanner>("ff", "activation_1");
    Handle<Computation> readB = makeObject<ff::FFMatrixScanner>("ff", "gradient_2");

    // make the join
    Handle<Computation> join = makeObject<ff::FFJoinBackTransposeMult>(embedding_size / embedding_block);
    join->setInput(0, readA);
    join->setInput(1, readB);

    // make the aggregation
    Handle<Computation> myAggregation = makeObject<ff::FFAggMatrix>();
    myAggregation->setInput(join);

    // make the writer
    Handle<Computation> myWriter = makeObject<ff::FFMatrixWriter>("ff", "d_w2");
    myWriter->setInput(myAggregation);

    // run the computation
    bool success = pdbClient.executeComputations({myWriter});
  }
  stage_end = std::chrono::steady_clock::now();
  std::cout << "Run d_w2 for " << std::chrono::duration_cast<std::chrono::nanoseconds>(stage_end - stage_begin).count() << "[ns]" << '\n';

  // calculate the gradient_1
  stage_begin = std::chrono::steady_clock::now();
  {
    // do the activation of the first layer
    const UseTemporaryAllocationBlock tempBlock{1024 * 1024 * 128};

    // make the computation
    Handle<Computation> readA = makeObject<ff::FFMatrixScanner>("ff", "gradient_2");
    Handle<Computation> readB = makeObject<ff::FFMatrixScanner>("ff", "w2");

    // make the join
    Handle<Computation> join = makeObject<ff::FFJoinBackMultTranspose>();
    join->setInput(0, readA);
    join->setInput(1, readB);

    // make the aggregation
    Handle<Computation> myAggregation = makeObject<ff::FFAggMatrix>();
    myAggregation->setInput(join);

    // make the writer
    Handle<Computation> myWriter = makeObject<ff::FFMatrixWriter>("ff", "gradient_1_tmp");
    myWriter->setInput(myAggregation);

    // run the computation
    bool success = pdbClient.executeComputations({myWriter});
  }
  stage_end = std::chrono::steady_clock::now();
  std::cout << "Run gradient_1_tmp for " << std::chrono::duration_cast<std::chrono::nanoseconds>(stage_end - stage_begin).count() << "[ns]" << '\n';

  // remove the gradient_2
  pdbClient.removeSet("ff", "gradient_2");

  // calculate the elementvise
  stage_begin = std::chrono::steady_clock::now();
  {
    // do the activation of the first layer
    const UseTemporaryAllocationBlock tempBlock{1024 * 1024 * 128};

    // make the computation
    Handle<Computation> readA = makeObject<ff::FFMatrixScanner>("ff", "gradient_1_tmp");
    Handle<Computation> readB = makeObject<ff::FFMatrixScanner>("ff", "activation_1");

    // make the join
    Handle<Computation> join = makeObject<ff::FFGradientJoin>();
    join->setInput(0, readA);
    join->setInput(1, readB);

    // make the aggregation
    Handle<Computation> myAggregation = makeObject<ff::FFAggMatrix>();
    myAggregation->setInput(join);

    // make the writer
    Handle<Computation> myWriter = makeObject<ff::FFMatrixWriter>("ff", "gradient_1");
    myWriter->setInput(myAggregation);

    // run the computation
    bool success = pdbClient.executeComputations({myWriter});
  }
  stage_end = std::chrono::steady_clock::now();
  std::cout << "Run gradient_1 for " << std::chrono::duration_cast<std::chrono::nanoseconds>(stage_end - stage_begin).count() << "[ns]" << '\n';

  // remove the activation_1
  pdbClient.removeSet("ff", "activation_1");
  pdbClient.removeSet("ff", "gradient_1_tmp");

  // calculate dw1
  stage_begin = std::chrono::steady_clock::now();
  {
    // do the activation of the first layer
    const UseTemporaryAllocationBlock tempBlock{1024 * 1024 * 128};

    // make the computation
    Handle<Computation> readA = makeObject<ff::FFMatrixScanner>("ff", "input_batch");
    Handle<Computation> readB = makeObject<ff::FFMatrixScanner>("ff", "gradient_1");

    // make the join
    Handle<Computation> join = makeObject<ff::FFJoinBackTransposeMult>(num_features / features_block);
    join->setInput(0, readA);
    join->setInput(1, readB);

    // make the aggregation
    Handle<Computation> myAggregation = makeObject<ff::FFAggMatrix>();
    myAggregation->setInput(join);

    // make the writer
    Handle<Computation> myWriter = makeObject<ff::FFMatrixWriter>("ff", "d_w1");
    myWriter->setInput(myAggregation);

    // run the computation
    bool success = pdbClient.executeComputations({myWriter});
  }
  stage_end = std::chrono::steady_clock::now();
  std::cout << "Run dw1 for " << std::chrono::duration_cast<std::chrono::nanoseconds>(stage_end - stage_begin).count() << "[ns]" << '\n';

  pdbClient.removeSet("ff", "gradient_1");

  // do the update for w1
  stage_begin = std::chrono::steady_clock::now();
  {
    // do the activation of the first layer
    const UseTemporaryAllocationBlock tempBlock{1024 * 1024 * 128};

    // make the computation
    Handle<Computation> readA = makeObject<ff::FFMatrixScanner>("ff", "w1");
    Handle<Computation> readB = makeObject<ff::FFMatrixScanner>("ff", "d_w1");

    // make the join
    Handle<Computation> join = makeObject<ff::FFUpdateJoin>();
    join->setInput(0, readA);
    join->setInput(1, readB);

    // make the aggregation
    Handle<Computation> myAggregation = makeObject<ff::FFAggMatrix>();
    myAggregation->setInput(join);

    // make the writer
    Handle<Computation> myWriter = makeObject<ff::FFMatrixWriter>("ff", "w1_updated");
    myWriter->setInput(myAggregation);

    // run the computation
    bool success = pdbClient.executeComputations({myWriter});
  }
  stage_end = std::chrono::steady_clock::now();
  std::cout << "Run update w1 for " << std::chrono::duration_cast<std::chrono::nanoseconds>(stage_end - stage_begin).count() << "[ns]" << '\n';

  // remove the w1 set
  pdbClient.removeSet("ff", "w1");
  pdbClient.removeSet("ff", "d_w1");

  // do the update w2
  stage_begin = std::chrono::steady_clock::now();
  {
    // do the activation of the first layer
    const UseTemporaryAllocationBlock tempBlock{1024 * 1024 * 128};

    // make the computation
    Handle<Computation> readA = makeObject<ff::FFMatrixScanner>("ff", "w2");
    Handle<Computation> readB = makeObject<ff::FFMatrixScanner>("ff", "d_w2");

    // make the join
    Handle<Computation> join = makeObject<ff::FFUpdateJoin>();
    join->setInput(0, readA);
    join->setInput(1, readB);

    // make the aggregation
    Handle<Computation> myAggregation = makeObject<ff::FFAggMatrix>();
    myAggregation->setInput(join);

    // make the writer
    Handle<Computation> myWriter = makeObject<ff::FFMatrixWriter>("ff", "w2_updated");
    myWriter->setInput(myAggregation);

    // run the computation
    bool success = pdbClient.executeComputations({myWriter});
  }
  stage_end = std::chrono::steady_clock::now();
  std::cout << "Run update w2 for " << std::chrono::duration_cast<std::chrono::nanoseconds>(stage_end - stage_begin).count() << "[ns]" << '\n';

  std::chrono::steady_clock::time_point planner_end = std::chrono::steady_clock::now();
  std::cout << "Run ff for " << std::chrono::duration_cast<std::chrono::nanoseconds>(planner_end - planner_begin).count()
            << "[ns]" << '\n';

  // remove the w1 set
  pdbClient.removeSet("ff", "w2");
  pdbClient.removeSet("ff", "d_w2");

  /// 5. Get the set from the

  // grab the iterator
  auto it = pdbClient.getSetIterator<ff::FFMatrixBlock>("ff", "w2_updated");
  int32_t count = 0;
  while (it->hasNextRecord()) {

    // grab the record
    auto r = it->getNextRecord();
    count++;
/*
    std::cout << "(" << r->getRowID() << ", " << r->getColID() << ")\n";
    // write out the values
    float *values = r->data->data->c_ptr();
    for (int i = 0; i < r->data->numRows; ++i) {
      for (int j = 0; j < r->data->numCols; ++j) {
        std::cout << values[i * r->data->numCols + j] << ", ";
      }
      std::cout << "\n";
    }

    if(r->data->bias != nullptr) {

      //
      std::cout << "Bias\n";
      for (int j = 0; j < r->data->numCols * r->data->numRows; ++j) {
        std::cout << values[j] << ", ";
      }
      std::cout << "\n";
    }
    std::cout << "\n\n"; */
  }

  // wait a bit before the shutdown
  sleep(4);

  std::cout << count << '\n';

  return 0;
}
