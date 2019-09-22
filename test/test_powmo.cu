#include <gtest/gtest.h>
#include "../src/powm_odd.cu"
int main() {
  typedef powm_params_t<8, 1024, 5> params;
  run_test<params>(128);
  typedef powm_params_t<32, 32768, 16> params_puzzle;
  /* run_simple_test<params_puzzle>(2, 15000); */
  /* run_simple_test<params_puzzle>(2, 15000); */
  /* run_simple_test<params_puzzle>(2, 15000); */
  /* run_simple_test<params_puzzle>(2, 15000); */

  run_simple_test<params_puzzle>(2, 2);
  run_simple_test<params_puzzle>(2, 4);
  run_simple_test<params_puzzle>(2, 8);
  run_simple_test<params_puzzle>(2, 16);
  run_simple_test<params_puzzle>(2, 32);
  run_simple_test<params_puzzle>(2, 64);
  run_simple_test<params_puzzle>(2, 128);
  run_simple_test<params_puzzle>(2, 256);
  run_simple_test<params_puzzle>(2, 512);
  run_simple_test<params_puzzle>(2, 1024);
  run_simple_test<params_puzzle>(2, 2048);
  run_simple_test<params_puzzle>(2, 4096);
  run_simple_test<params_puzzle>(2, 8192);
  /* run_simple_test<params_puzzle>(2, 16384); */
  /* run_simple_test<params_puzzle>(2, 32768); */
  /* run_simple_test<params_puzzle>(2, 65536); */

  /* run_puzzle_test<params_puzzle>(100, 15000); */
  return 0;
}
