#include <mkl.h>
#include <iostream>
#include <chrono>

int main() {

  // get the sizes
  uint32_t I = 10000;
  uint32_t J = 10000;
  uint32_t K = 10000;


  // get the ptrs
  auto *outData = new float[10000 * 10000];
  auto *in1Data = new float[10000 * 10000];
  auto *in2Data = new float[10000 * 10000];

  for(int i = 0; i < I; i++) {
    for(int j = 0; j < J; j++) {
      in1Data[i * J + j] = i + j + 1;
      in2Data[i * J + j] = i + j + 2;
    }
  }

  std::chrono::steady_clock::time_point planner_begin = std::chrono::steady_clock::now();

  // do the multiply
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, I, J, K, 1.0f, in1Data, K, in2Data, J, 0.0f, outData, J);

  std::chrono::steady_clock::time_point planner_end = std::chrono::steady_clock::now();
  std::cout << "Run multiply for " << std::chrono::duration_cast<std::chrono::nanoseconds>(planner_end - planner_begin).count()
            << "[ns]" << '\n';
}