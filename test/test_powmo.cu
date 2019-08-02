#include <gtest/gtest.h>
#include <time.h>
#include "../src/powm_odd.cu"
int main() {
  typedef powm_params_t<8, 1024, 5> params;
  printf("[%s]: Running normal test...\n", time(NULL));
  run_test<params>(10000);
  typedef powm_params_t<16, 4096, 5> params_puzzle;
  printf("[%s]: Running puzzle test...\n", time(NULL));
  run_puzzle_test<params_puzzle>(10000, 1024);
  return 0;
}
