#include <gtest/gtest.h>
#include "../src/powm_odd.cu"
int main() {
  typedef powm_params_t<8, 1024, 5> params;
  run_test<params>(128);
  typedef powm_params_t<32, 32768, 16> params_puzzle;
  run_simple_test<params_puzzle>(2, 15000);
  run_simple_test<params_puzzle>(2, 15000);
  run_simple_test<params_puzzle>(2, 15000);
  run_simple_test<params_puzzle>(2, 15000);
  run_puzzle_test<params_puzzle>(100, 15000);
  return 0;
}
