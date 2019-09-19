#include <gtest/gtest.h>
#include "../src/powm_odd.cu"
int main() {
  typedef powm_params_t<8, 1024, 5> params;
  run_test<params>(128);
  typedef powm_params_t<16, 32768, 5> params_puzzle;
  run_puzzle_test<params_puzzle>(100, 25000);
  return 0;
}
